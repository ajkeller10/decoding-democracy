import os
import json
import utils
from collections import OrderedDict
import xmltodict_master.xmltodict as xmltodict 

segment_path='input/ICSI_plus_NXT/ICSIplus/Segments/'
file_paths = os.listdir(segment_path)
print(file_paths)


paths = utils.get_paths('plus')

file_name_list = [f for f in os.listdir(segment_path) if f.endswith('.segs.xml')]

for file_name in file_name_list:
    meeting_id = utils.get_meeting_id(file_name)

    # load dialogueActs
    dialogueActs = {}
    dialogueAct_json = json.load(open('output/dialogueActs/' + meeting_id + '.json'))
    for dialogueAct in dialogueAct_json:
        dialogueActs[dialogueAct['id']] = dialogueAct

    # load segments
    segments = OrderedDict()
    segments_xml = utils.xml_to_dict(
        segment_path + meeting_id + letter +'.segs.xml', #stopping here 
        force_list=('sentence',)
    )
    for type in ['abstract', 'decisions', 'problems', 'progress']:
        if 'sentence' not in segments_xml['nite:root'][type].keys():
            print(str(meeting_id) + ": Missing " + type + " sentence")
            continue

        for segments_sentence in segments_xml['nite:root'][type]['sentence']:
            segments_sentence_id = segments_sentence['@nite:id']
            segments_sentence_text = segments_sentence['#text']

            segments[segments_sentence_id] = {
                'id': segments_sentence_id,
                'text': segments_sentence_text,
                'type': type
            }

    # load summlink
    summlinks = {}
    summlink_xml = utils.xml_to_dict(
        paths['extractive'] + file_name,
        force_list=()
    )

    for summlink in summlink_xml['nite:root']['summlink']:
        id = summlink['@nite:id']

        if summlink['nite:pointer'][0]['@role']=='extractive' and summlink['nite:pointer'][1]['@role']=='segments':
            extractive_id = summlink['nite:pointer'][0]['@href'].split('#')[1].split('(')[1].split(')')[0]
            segments_id = summlink['nite:pointer'][1]['@href'].split('#')[1].split('(')[1].split(')')[0]

            try:
                summlinks[segments_id].append(dialogueActs[extractive_id])
            except:
                summlinks[segments_id] = [dialogueActs[extractive_id]]
        else:
            raise RuntimeError()

    summlinks_output = []

    for segments_id in sorted(summlinks.keys(), key = lambda x: int(x.split('.')[-1])):
        summlinks_output.append({
            'segments': segments[segments_id],
            'extractive': summlinks[segments_id]
        })

    output_dir = 'output/summlink/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_dir+meeting_id+'.json', 'w') as f:
        json.dump(summlinks_output, f)

