import os
import shutil

for track_name in os.listdir('tracks'):
    for stem in ['bass', 'drums', 'other', 'vocals']:
        file_name = f'{stem}.wav'
        file_path = os.path.join('tracks', track_name, file_name)
        if not os.path.exists(os.path.join(file_path)):
            break
        out_path = os.path.join('stems', stem, track_name)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        shutil.move(file_path,
   	            os.path.join(out_path, file_name))

