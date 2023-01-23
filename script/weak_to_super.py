import os
import shutil
from pathlib import Path

split_ranges = {'train': (1, 1500), 'validation': (1501, 1875), 'test': (1876, 2098)}


def main():
    for split in ['test']:
        split_start, split_end = split_ranges[split]
        split_dir = Path('/data/Slakh_supervised') / split
        os.mkdir(split_dir)
        for n in range(split_start, split_end + 1):
            base_out_path = split_dir / f'Track{n:05}'
            os.mkdir(base_out_path)
            for stem in ['bass', 'drums', 'guitar', 'piano']:
                in_path = os.path.join('/data/Slakh', split, f'{stem}' , f'Track{n:05}.wav')
                print(in_path, os.path.exists(in_path))

                if os.path.exists(in_path):
                    out_path = base_out_path / f'{stem}.wav'
                    shutil.copy(in_path, out_path)
        # remove empty dirs
        for n in range(split_start, split_end + 1):
            track_dir = split_dir / f'Track{n:05}'
            if not any(track_dir.iterdir()):
                os.rmdir(track_dir)


if __name__ == '__main__':
    main()
