"""Split a recommendation dataset into independently trainable user groups.

Users are assigned using the number of non-zero interactions in trn_mat.pkl.
User rows and user-side embeddings are compacted in each output dataset, while
the original item ID space and item-side embeddings are retained.
"""

import argparse
import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse


MATRIX_FILES = ('trn_mat.pkl', 'val_mat.pkl', 'tst_mat.pkl')
USER_EMBEDDING_FILES = ('usr_emb_np.pkl', 'user_intent_emb_3.pkl')
ITEM_EMBEDDING_FILES = ('itm_emb_np.pkl', 'item_intent_emb_3.pkl')


@dataclass(frozen=True)
class UserGroup:
    suffix: str
    label: str
    minimum: int
    maximum: Optional[int]

    def select(self, interaction_counts):
        selected = interaction_counts >= self.minimum
        if self.maximum is not None:
            selected &= interaction_counts <= self.maximum
        return selected


GROUPS = (
    UserGroup('u1_5', '1-5', 1, 5),
    UserGroup('u6_10', '6-10', 6, 10),
    UserGroup('u11_plus', '>=11', 11, None),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split users by training-set interaction count.')
    parser.add_argument(
        '--dataset', default='movie',
        help='Source directory name under data/ (default: movie).')
    parser.add_argument(
        '--data-root', type=Path, default=None,
        help='Data root (default: <repository>/data).')
    parser.add_argument(
        '--copy-item-embeddings', action='store_true',
        help='Copy item embeddings instead of space-saving hard links.')
    return parser.parse_args()


def load_sparse_matrix(path):
    with path.open('rb') as f:
        matrix = sparse.csr_matrix(pickle.load(f))
    matrix.eliminate_zeros()
    return matrix


def dump_pickle(value, path):
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    with temporary_path.open('wb') as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, path)


def link_or_copy(source, destination, copy_file):
    if copy_file:
        shutil.copy2(source, destination)
        return 'copy'
    try:
        os.link(source, destination)
        return 'hardlink'
    except OSError:
        shutil.copy2(source, destination)
        return 'copy'


def validate_embedding(array, expected_rows, path):
    if not isinstance(array, np.ndarray):
        raise TypeError('{} must contain a NumPy array'.format(path))
    if array.ndim == 0 or array.shape[0] != expected_rows:
        raise ValueError(
            '{} has shape {}; expected first dimension {}'.format(
                path, array.shape, expected_rows))


def main():
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root or repository_root / 'data'
    source_dir = data_root / args.dataset
    if not source_dir.is_dir():
        raise FileNotFoundError('Dataset directory does not exist: {}'.format(source_dir))

    matrices = {
        filename: load_sparse_matrix(source_dir / filename)
        for filename in MATRIX_FILES
    }
    train_matrix = matrices['trn_mat.pkl']
    user_num, item_num = train_matrix.shape
    for filename, matrix in matrices.items():
        if matrix.shape != train_matrix.shape:
            raise ValueError(
                '{} has shape {}; expected {}'.format(
                    filename, matrix.shape, train_matrix.shape))

    interaction_counts = np.diff(train_matrix.indptr)
    selections = {group.suffix: group.select(interaction_counts) for group in GROUPS}
    coverage = sum(mask.astype(np.int8) for mask in selections.values())
    if not np.all(coverage == 1):
        raise ValueError(
            'Groups must cover every user exactly once; training rows with zero '
            'interactions are not supported.')

    output_dirs = {
        group.suffix: data_root / '{}_{}'.format(args.dataset, group.suffix)
        for group in GROUPS
    }
    existing = [str(path) for path in output_dirs.values() if path.exists()]
    if existing:
        raise FileExistsError(
            'Refusing to overwrite existing output directories: {}'.format(
                ', '.join(existing)))
    for path in output_dirs.values():
        path.mkdir(parents=False)

    group_user_ids = {
        suffix: np.flatnonzero(mask).astype(np.int64)
        for suffix, mask in selections.items()
    }

    try:
        for filename, matrix in matrices.items():
            for group in GROUPS:
                user_ids = group_user_ids[group.suffix]
                grouped_matrix = matrix[user_ids].tocoo()
                dump_pickle(grouped_matrix, output_dirs[group.suffix] / filename)

        for filename in USER_EMBEDDING_FILES:
            source_path = source_dir / filename
            with source_path.open('rb') as f:
                embedding = pickle.load(f)
            validate_embedding(embedding, user_num, source_path)
            for group in GROUPS:
                user_ids = group_user_ids[group.suffix]
                dump_pickle(
                    embedding[user_ids], output_dirs[group.suffix] / filename)

        item_file_modes = {}
        for filename in ITEM_EMBEDDING_FILES:
            source_path = source_dir / filename
            with source_path.open('rb') as f:
                embedding = pickle.load(f)
            validate_embedding(embedding, item_num, source_path)
            del embedding
            for group in GROUPS:
                mode = link_or_copy(
                    source_path,
                    output_dirs[group.suffix] / filename,
                    args.copy_item_embeddings)
                item_file_modes.setdefault(group.suffix, {})[filename] = mode

        known_files = set(MATRIX_FILES + USER_EMBEDDING_FILES + ITEM_EMBEDDING_FILES)
        for source_path in source_dir.iterdir():
            if not source_path.is_file() or source_path.name in known_files:
                continue
            for group in GROUPS:
                link_or_copy(
                    source_path,
                    output_dirs[group.suffix] / source_path.name,
                    args.copy_item_embeddings)

        for group in GROUPS:
            output_dir = output_dirs[group.suffix]
            user_ids = group_user_ids[group.suffix]
            np.save(output_dir / 'original_user_ids.npy', user_ids, allow_pickle=False)
            split_stats = {}
            for filename, matrix in matrices.items():
                grouped_matrix = matrix[user_ids]
                row_counts = np.diff(grouped_matrix.indptr)
                split_stats[filename] = {
                    'interactions': int(grouped_matrix.nnz),
                    'users_with_interactions': int(np.count_nonzero(row_counts)),
                }
            metadata = {
                'source_dataset': args.dataset,
                'group': group.label,
                'train_interaction_min': group.minimum,
                'train_interaction_max': group.maximum,
                'user_count': int(user_ids.size),
                'item_count': int(item_num),
                'source_user_count': int(user_num),
                'split_stats': split_stats,
                'item_embedding_storage': item_file_modes[group.suffix],
                'user_id_mapping': 'original_user_ids.npy',
            }
            with (output_dir / 'split_meta.json').open('w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                f.write('\n')

            print(
                '{}: users={}, train={}, validation={}, test={}'.format(
                    output_dir.name,
                    user_ids.size,
                    split_stats['trn_mat.pkl']['interactions'],
                    split_stats['val_mat.pkl']['interactions'],
                    split_stats['tst_mat.pkl']['interactions']))
    except Exception:
        for output_dir in output_dirs.values():
            if output_dir.exists():
                shutil.rmtree(output_dir)
        raise


if __name__ == '__main__':
    main()
