import os
import shutil

split_ranges = {'train': (1, 1500),
                'validation': (1501, 1875),
                'test': (1876, 2098)}


def main():
    for split in ['train', 'validation', 'test']:
        split_range = split_ranges[split]
        for n in range(split_range[0], split_range[1] + 1):
            base_out_path = os.path.join('SLAKH2000_by_track', split, f'Track{n:05}')
            os.mkdir(base_out_path)
            for stem in ['bass', 'drums', 'guitar', 'piano']:
                in_path = os.path.join('SLAKH2000', f'{stem}_22050', split, f'Track{n:05}.wav')
                if os.path.exists(in_path):
                    out_path = os.path.join(base_out_path, f'{stem}.wav')
                    shutil.move(in_path, out_path)


if __name__ == '__main__':
    main()
