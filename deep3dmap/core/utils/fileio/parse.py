# Copyright (c) achao2013. All rights reserved.
def list_from_file(filename, prefix='', offset=0, max_num=0, encoding='utf-8'):
    """Load a text file and parse the content as a list of strings.

    Args:
        filename (str): Filename.
        prefix (str): The prefix to be inserted to the beginning of each item.
        offset (int): The offset of lines.
        max_num (int): The maximum number of lines to be read,
            zeros and negatives mean no limitation.
        encoding (str): Encoding used to open the file. Default utf-8.

    Returns:
        list[str]: A list of strings.
    """
    cnt = 0
    item_list = []
    with open(filename, 'r', encoding=encoding) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n\r'))
            cnt += 1
    return item_list


def dict_from_file(filename, key_type=str):
    """Load a text file and parse the content as a dict.

    Each line of the text file will be two or more columns split by
    whitespaces or tabs. The first column will be parsed as dict keys, and
    the following columns will be parsed as dict values.

    Args:
        filename(str): Filename.
        key_type(type): Type of the dict keys. str is user by default and
            type conversion will be performed if specified.

    Returns:
        dict: The parsed contents.
    """
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            items = line.rstrip('\n').split()
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping
