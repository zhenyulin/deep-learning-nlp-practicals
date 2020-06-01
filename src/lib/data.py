from lxml import etree
from random import shuffle
from tqdm import tqdm
import numpy as np
import os
import urllib.request
import zipfile


def download_datafile(filename, url, project_path="."):
    data_file = f"{project_path}/data/{filename}"
    if not os.path.isfile(data_file):
        print(f"{filename} download start...")
        urllib.request.urlretrieve(url, filename=data_file)
        print(f"{filename} download completed.")
    else:
        print(f"{filename} already downloaded.")
    return data_file


def load_TED_dataset(include_keywords=False, include_summary=False, project_path="."):
    with zipfile.ZipFile(
        download_datafile(
            "ted_en-20160408.zip",
            "https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip",
            project_path,
        ),
        "r",
    ) as z:
        unzipped = z.open("ted_en-20160408.xml", "r")
        doc = etree.parse(unzipped)
        content_text = doc.xpath("//content/text()")
        keywords_text = doc.xpath("//keywords/text()")
        summary_text = doc.xpath("//description/text()")

        del unzipped
        del doc

        if include_keywords and include_summary:
            return content_text, keywords_text, summary_text

        if include_keywords:
            return content_text, keywords_text

        if include_summary:
            return content_text, summary_text

        return content_text


def drop_data_of_label(dataset, target, drop_number):
    output = []
    counter = 0
    for entry in dataset:
        if entry[1] == target and counter < drop_number:
            counter += 1
        else:
            output.append(entry)
    return output


def shuffle_take_data(data, portion=0.2):
    copied = data.copy()
    shuffle(copied)
    data_sample = copied[: int(len(copied) * portion)]
    print(f"sampled {len(data_sample)}/{len(copied)}")
    return data_sample


def batch_pad_truncate_dataset(encoded_dataset, batch_size, truncate_length):
    longest = max([len(clip) for clip in encoded_dataset])
    # we want a +1*truncate_length padding even for longest clip
    num_truncate = (longest // truncate_length) + 1
    # match it to integer times truncate length
    full_length = num_truncate * truncate_length
    no_truncate_batch_num = len(encoded_dataset) // batch_size

    padded_dataset = [
        np.append(clip, np.zeros(full_length - len(clip), dtype=int))
        for clip in encoded_dataset
    ]

    output = [
        clip[m * truncate_length : (m + 1) * truncate_length]
        for n in tqdm(range(no_truncate_batch_num))
        for m in range(num_truncate)
        for clip in padded_dataset[n * batch_size : (n + 1) * batch_size]
    ]

    return output


def shuffle_split_dataset(dataset, validation_test_size=0, do_shuffle=True):
    copied = dataset.copy()

    size = len(dataset) // 10 if validation_test_size == 0 else validation_test_size

    if do_shuffle:
        shuffle(copied)

    contents = np.array([content for (content, label) in copied])
    labels = np.array([label for (content, label) in copied])

    test_set = (contents[:size], labels[:size])
    validation_set = (
        contents[size : size * 2],
        labels[size : size * 2],
    )
    training_set = (contents[size * 2 :], labels[size * 2 :])

    return training_set, validation_set, test_set
