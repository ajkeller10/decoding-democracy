FILLERS = ["um", "uh", "oh", "hmm", "mm-hmm", "uh-uh", "you know"]


def preprocessing(df, caption_col_name="caption", min_caption_len=20):
    """Strips filler words and deletes sentences with 20 characters or less."""
    fillers += list(
        map(lambda filler: filler + " ", fillers)
    )  # filler inside caption with other words
    fillers = list(
        map(lambda filler: "(?i)" + filler, fillers)
    )  # make it case-insensitive
    df[caption_col_name].replace(fillers, [""] * len(fillers), regex=True, inplace=True)
    df[caption_col_name].replace('<([^<>]+)>', "", regex=True, inplace=True)

    # divide up multi-sentence captions into new rows
    df[caption_col_name] = df[caption_col_name].str.split(".").explode()

    df = df[df[caption_col_name].str.len() > min_caption_len]
    df.reset_index(inplace=True)

    return df


def icsi_dataset():
    """This data was mostly parsed from the NTX tool, read more googling ICSI Meeting Corpus

    It is stored in a internal database and can be accessed with the following query

    input_df = 
            SELECT
                meeting_id,
                st,
                en,
                caption,
                speaker
            FROM {icsi}
            WHERE ds = '2021-01-12'

    label_df = SELECT
                fb_meeting_id AS meeting_id,
                st,
                en,
                topic
            FROM {labels}
            WHERE ds = '2021-01-10'

    label_df
    """
    raise NotImplementedError("Need to download dataset and set up dataset.py.")
    return input_df, label_df


def ami_dataset():
    """See XXXX for label generation and XXXX for input analysis

            SELECT
                fb_meeting_id AS meeting_id,
                st,
                en,
                caption,
                speaker
            FROM {ami}
            WHERE ds = '2021-01-12'

            SELECT
                fb_meeting_id AS meeting_id,
                st,
                en,
                topic
            FROM {labels}
            WHERE ds = '2021-01-10'
    """
    raise NotImplementedError("Need to download dataset and set up dataset.py.")
    return input_df, label_df


