from pathlib import Path
import shutil

root_dir = Path('data/Slakh/test/')  # Change this to the path of your root directory

# Iterate through the subfolders
for subdir in root_dir.iterdir():
    if subdir.is_dir():
        for file in subdir.iterdir():
            if file.suffix != ".dur": 
                # Get the instrument name and track name
                instrument_name = subdir.name
                track_name = file.stem

                # Create the new track folder if it doesn't exist
                new_root_dir = root_dir.parent.parent / "Slakh_track_first" / root_dir.name
                new_track_folder = new_root_dir / track_name
                new_track_folder.mkdir(exist_ok=True, parents=True)

                # Move the file to the new track folder and rename it
                new_file_path = new_track_folder / (instrument_name + file.suffix)
                shutil.copy(str(file), str(new_file_path))

# # Remove the old instrument folders
# for subdir in root_dir.iterdir():
#     if subdir.is_dir() and subdir.name not in ['bass', 'drums', 'guitar', 'piano']:
#         shutil.rmtree(str(subdir))
