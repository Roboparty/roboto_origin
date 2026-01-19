# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
from pathlib import Path

"""This script can be used to rename the template project to a new project name.
It renames all the occurrences of ext_template (in files, directories, etc.) to the new project name.
"""


def rename_file_contents(root_dir_path: str, old_name: str, new_name: str, exclude_dirs: list = []):
    """Rename all instances of the old keyword to the new keyword in all files in the root directory.

    Args:
        root_dir_path (str): The root directory path.
        old_name (str): The old keyword to replace.
        new_name (str): The new keyword to replace with.
    """
    for dirpath, _, files in os.walk(root_dir_path):
        if any(exclude_dir in dirpath for exclude_dir in exclude_dirs):
            continue
        for file_name in files:
            if file_name == "rename_template.py":
                continue
            try:
                with open(os.path.join(dirpath, file_name)) as file:
                    file_contents = file.read()
                file_contents = file_contents.replace(old_name, new_name)
                with open(os.path.join(dirpath, file_name), "w") as file:
                    file.write(file_contents)
            except Exception as e:
                print(f"Ignoring {file_name}: {e}")
                continue


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_template.py <new_name>")
        sys.exit(1)

    root_dir_path = str(Path(__file__).resolve().parent.parent)
    old_name = "ext_template"
    new_name = sys.argv[1]

    print(f"Warning, this script will rename all instances of '{old_name}' to '{new_name}' in {root_dir_path}.")
    proceed = input("Proceed? (y/n): ")

    if proceed.lower() == "y":
        # rename the ext_template folder
        os.rename(
            os.path.join(root_dir_path, "source", "ext_template", "ext_template"),
            os.path.join(root_dir_path, "source", "ext_template", new_name),
        )
        os.rename(
            os.path.join(root_dir_path, "source", "ext_template"), os.path.join(root_dir_path, "source", new_name)
        )
        # rename the file contents
        rename_file_contents(root_dir_path, old_name, new_name, exclude_dirs=[".git", "logs"])
    else:
        print("Aborting.")
