"""Update the data with missing labels or wrong labels at a folder level."""
from argparse import ArgumentParser
from re import search
from re import I
import os


valid_ner_tags = ['NEN', 'NEO', 'NEAR', 'NETI', 'NEL', 'NEP']


def read_lines_from_file(file_path):
    """Read lines from file with blanks."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return file_read.readlines()


def write_lines_to_file(lines, file_path):
    """Write lines to a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines) + '\n')


def update_lines_in_labeled_lines(lines):
    """Update lines in labeled lines by handling errors."""
    updated_lines = []
    for line in lines:
        line = line.strip()
        if line:
            line_split = line.split('\t')
            line_split = line_split[: 2]
            if len(line_split) <= 1:
                continue
            else:
                line_split[1] = line_split[1].replace('\u200c', '')
                if line_split[1] == '-':
                    line_split[1] = 'O'
                    updated_line = '\t'.join(line_split)
                elif line_split[1].startswith('-') and len(line_split[1]) > 1:
                    line_split[1] = 'B' + line_split[1]
                    updated_line = '\t'.join(line_split)
                elif line_split[1][0] not in ['B', 'I', 'O']:
                    line_split[1] = 'O'
                    updated_line = '\t'.join(line_split)
                elif search('NEMI|NEU', line_split[1], I):
                    line_split[1] = 'O'
                    updated_line = '\t'.join(line_split)
                else:
                    if line_split[1] != 'O':
                        tag_type, tag = line_split[1].split('-')
                        tag = tag.upper()
                        if tag not in valid_ner_tags or tag_type not in ['B', 'I']:
                            print(tag, tag_type)
                            tag = 'O'
                        else:
                            tag = line_split[1]
                        updated_line = '\t'.join([line_split[0], tag])
                    else:
                        updated_line = line
        else:
            updated_line = line
        updated_lines.append(updated_line)
    return updated_lines


def read_files_update_and_write_to_files(input_folder_path, output_folder_path):
    """Read files from a folder, update them in case of errors, and write them into another file in a folder."""
    for root, dirs, files in os.walk(input_folder_path):
        for fl in files:
            input_path = os.path.join(root, fl)
            language = search('^(.*?)\-.*?\.txt', fl).group(1)
            print(fl)
            language_folder_path = os.path.join(output_folder_path, language)
            if not os.path.isdir(language_folder_path):
                os.makedirs(language_folder_path)
            input_lines = read_lines_from_file(input_path)
            updated_lines = update_lines_in_labeled_lines(input_lines)
            output_path = os.path.join(language_folder_path, fl)
            write_lines_to_file(updated_lines, output_path)


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser()
    parser.add_argument('--input', dest='inp', help='Enter the input folder path')
    parser.add_argument('--output', dest='out', help='Enter the output folder path')
    args = parser.parse_args()
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    read_files_update_and_write_to_files(args.inp, args.out)


if __name__ == '__main__':
    main()
