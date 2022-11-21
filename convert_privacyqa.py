import itertools
from os import PathLike
from pathlib import Path
import random
import secrets
from typing import Any, Optional

import click
import json
import pandas as pd


class Record:
    def __init__(self, folder: str, query: str, segment: str, label: str):
        self.folder = folder
        self.query = query
        self._segment = segment
        self.label = label
    
    @property
    def clean_folder(self):
        return self.folder.replace("../../Dataset/Train/", "")
    
    @property
    def segment(self):
        if not self._segment.rstrip().endswith((".", ":")):
            return self._segment.rstrip() + "."
        else:
            return self._segment.rstrip()
        
    @property
    def clean_segment(self):
        if not self._segment.strip().endswith((".", ":")):
            return self._segment.strip() + "."
        else:
            return self._segment.strip()
        
    @property
    def clean_query(self):
        return self.query.strip()

@click.command
@click.argument("data_dir", required=True, type=str)
# @click.option("--v2", required=True, type=str)
def convert(data_dir: PathLike):
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise ValueError("Specified data dir does not exist")
    
    
    data = {'train': {}, 'test': {}}
    
    for file in data_dir.iterdir():
        if file.is_file() and file.suffix == '.csv':
            if 'train' in file.name:
                data['train']['data'] = file
            elif 'test' in file.name:
                data['test']['data'] = file
        # Not sure if meta-annotations is necessary, leave out for now
    
    output = {
        "version": "v1.0",
        "data": [
        ]
    }
    
    counter = itertools.count(start=0, step=1)
    
    for key in ['train', 'test']:
        data_pre = pd.read_csv(data[key]['data'], delimiter='\t').drop(['DocID', 'QueryID', 'SentID', 'Split'], axis=1)
        
        data_pre['Folder'] = data_pre['Folder'].apply(lambda x: x.replace(f"../../Dataset/{key.capitalize()}/", ""))
        
        data_dict =data_pre.to_dict(orient='records')
        
        # Need to preprocess test data
        if key == 'test':
            for record in data_dict:
                count_relevant = 0
                count_irrelevant = 0 
                count_none = 0
                for ann in ['Ann1','Ann2','Ann3','Ann4','Ann5','Ann6']:
                    if record[ann] == 'Relevant':
                        count_relevant += 1
                    if record[ann] == 'Irrelevant':
                        count_irrelevant += 1
                    if record[ann] == 'None':
                        count_none += 1
                    del record[ann]
                if (count_irrelevant ==0 and count_relevant ==0) or count_relevant < count_irrelevant:
                    record['Label'] = 'Irrelevant'
                elif count_relevant >= count_irrelevant:
                    record['Label'] = 'Relevant'
                
        
        all_records: list[Record] = []
        
        for record in data_dict:
            all_records.append(Record(record['Folder'], record['Query'], record['Segment'], record['Label']))
        
        lookup = set()  # a temporary lookup set
        uniq_fold = [record.clean_folder for record in all_records if record.clean_folder not in lookup and lookup.add(record.clean_folder) is None]
        
        del lookup
        for fold in uniq_fold:
            print(fold)
            eph_records = [record for record in all_records if record.clean_folder == fold]
            
            fold_dict = {
                'title': fold,
                "paragraphs": [],
            }
            lookup = set()  # a temporary lookup set
            segments = [record.segment for record in eph_records if record.segment not in lookup and lookup.add(record.segment) is None]
            paras = list(paragraph_builder(segments))
            
            questions = list(set([record.clean_query for record in eph_records]))
            
            for para in paras:
                para_dict = {
                    "qas": [
                    ],
                    "index": next(counter),
                    "summary": [seg.strip() for seg in para.segments],
                    "context": para.context,
                }
                
                search_records = [record for record in eph_records if record.segment in para]
                for question in questions:
                    records = []
                    for segment in para.segments:
                        records.append(get_record(search_records, question, segment))
                    
                    relevant = [record for record in records if record.label == 'Relevant']
                    # irrelevant = [record for record in records if record.label == 'Irrelevant'] # May use for v2.0
                    
                    if len(relevant) == 0:
                        # Skip, add flag later for unanswerable v2
                        pass               
                    if len(relevant) == 1:
                        question_dict = {
                            "question": question,
                            # "type": "First Party Collection/Use|||Collection Mode|||Explicit",
                            "id": str(secrets.token_hex(8)),
                            "answers": [{'text': relevant[0].clean_segment, 'answer_start': para.index_of(relevant[0].clean_segment)}]
                        }
                        para_dict['qas'].append(question_dict)
                    if len(relevant) > 1:
                        question_dict = {
                            "question": question,
                            # "type": "First Party Collection/Use|||Collection Mode|||Explicit",
                            "id": str(secrets.token_hex(8)),
                            "answers": [{'text': record.segment, 'answer_start': para.index_of(record.clean_segment)} for record in relevant]
                        }
                        para_dict['qas'].append(question_dict)
                if len(para_dict['qas']) > 0:
                    fold_dict['paragraphs'].append(para_dict)
            output['data'].append(fold_dict)
    
            with open(Path(f"{data_dir}/policy_{key}_squad.json"), mode='w') as out_file:
                json.dump(output, out_file, indent=4)
        

def get_record(records: list[Record], query: str, segment: str):
    for record in records:
        if record.clean_query in query and record.segment in segment: # Need to do `in` to get around bug with periods at end
            return record
    return None

def get_unique_segments(records: list[Record]):
    segments = []
    
    for record in records:
        if record.segment not in segments:
            segments.append(record.segment)
    return segments

class Paragraph:
    def __init__(self, start: int, initial_segment:Optional[str] = None):
        if initial_segment:
            self.segments = [initial_segment]
        else:
            self.segments = []
        self._start = start
        self._end = start
        
    def add_segment(self, sent):
        self.segments.append(sent)
        self._end += 1
        
    def has_segments(self):
        return self.segments is not []
    
    def index_of(self, sub):
        return self.context.index(sub)
    
    def __repr__(self):
        return "".join(self.segments)
    
    def __contains__(self, obj):
        for segment in self.segments:
            if obj in segment:
                return True
        return False
    
    @property
    def context(self):
        return "".join(self.segments)
    
def paragraph_builder(segments: list[str]) -> list[Paragraph]:
    tmp_para = Paragraph(0)
    for i, segment in enumerate(segments):
        if segment.startswith(" "):
            if not segment.strip().endswith((".", ":")):
                tmp_para.add_segment(segment + ".")
            else:
                tmp_para.add_segment(segment)
        else:
            if not tmp_para.has_segments:
                if not segment.strip().endswith((".", ":")):
                    tmp_para.add_segment(segment + ".")
                else:
                    tmp_para.add_segment(segment)
            else:
                yield tmp_para
                tmp_para = Paragraph(i, segment)
        

    
if __name__ == '__main__':
    convert()
    
"""
Ignore relevant data points that end in :

"""