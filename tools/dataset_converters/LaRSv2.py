"""Renumbers LaRS classes so that thing classes start at 1."""
import json
import argparse
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert LaRS COCO annotations to mmdet accepted.')
    parser.add_argument('dataset_root', help='LaRS dataset (split) root path')
    parser.add_argument('--annotation_file', default='panoptic_annotations.json', type=str)
    parser.add_argument('--output_file', default='mmdet_annotations.json', type=str)
    parser.add_argument('--splits', default=['train', 'val', 'test'], type=str, nargs='+')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    new_categories = None
    for split in args.splits:
        in_file = osp.join(args.dataset_root, split, args.annotation_file)
        with open(in_file, 'r') as file:
            data = json.load(file)

        # Reorder and renumber categories: only do this once so all splits have the same IDs.
        if new_categories is None:
            # Reorder - things first
            thing_categories, stuff_categories = [], []
            for cat in data['categories']:
                if cat['isthing'] == 1:
                    thing_categories.append(cat)
                else:
                    stuff_categories.append(cat)

            # Renumber categories (starting with 1)
            new_categories = []
            id_map = {}
            for i, cat in enumerate(thing_categories + stuff_categories):
                cur_id = i+1
                id_map[cat['id']] = cur_id
                cat['id'] = cur_id
                new_categories.append(cat)

        data['categories'] = new_categories

        # Correct annotations
        for ann in data['annotations']:
            for segment in ann['segments_info']:
                segment['category_id'] = id_map[segment['category_id']]


        out_file = osp.join(args.dataset_root, split, args.output_file)
        with open(out_file, 'w') as file:
            json.dump(data, file)

if __name__=='__main__':
    main()
