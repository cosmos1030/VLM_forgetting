#!/bin/bash
cd ovdeval
# Loop through all .tar files
for f in *.tar; do
    # Extract the base name (e.g., "celebrity" from "celebrity.tar")
    foldername="${f%.tar}"
    echo "Processing $f into folder $foldername"

    # Create the directory
    mkdir -p "$foldername" # -p creates parent directories if they don't exist

    # Extract the tar file into the new directory
    tar -xf "$f" -C "$foldername/"

    echo "Finished $f"
done

# # Loop through all .tar.gz files
# for f in *.tar.gz; do
#     # Extract the base name (e.g., "logo" from "logo.tar.gz")
#     foldername="${f%.tar.gz}"
#     echo "Processing $f into folder $foldername"

#     # Create the directory
#     mkdir -p "$foldername"

#     # First, decompress the .tar.gz to .tar
#     gunzip "$f"

#     # The file is now "$foldername.tar". Extract it.
#     tar -xf "${foldername}.tar" -C "$foldername/"

#     # Optionally, remove the .tar file after extraction if you no longer need it
#     rm "${foldername}.tar"

#     echo "Finished $f"
# done

echo "All files processed."