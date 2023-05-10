import sys

def preprocessing(df, caption_col_name):
    """Strips filler words and deletes sentences with 20 characters or less."""
    fillers = ["um", "uh", "oh", "hmm", "you know", "like"]  # Drew: they remove like? seems odd
    fillers += list(
        map(lambda filler: filler + " ", fillers)
    )  # filler inside caption with other words
    fillers = list(
        map(lambda filler: "(?i)" + filler, fillers)
    )  # make it case-insensitive
    df[caption_col_name].replace(fillers, [""] * len(fillers), regex=True, inplace=True)

    captions_with_multiple_sentences = len(df.loc[df[caption_col_name].isin(["."])])
    if captions_with_multiple_sentences > 0:
        print(
            f"WARNING: Found {captions_with_multiple_sentences} captions with multiple sentences; sentence embeddings may be inaccurate.",
            file=sys.stderr,
        )

    df = df[df[caption_col_name].str.len() > 20]
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


