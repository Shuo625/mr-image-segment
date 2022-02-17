import yaml
import pandas as pd


def read_yaml(yaml_file: str) -> dict:
    with open(yaml_file, mode='r') as f:
        cfg = yaml.safe_load(f)

    return cfg


def combine_submissions(submissions: dict, out_submission_file: str) -> None:
    rst = {}

    for submission, weight in submissions.items():
        submission_dataframe = pd.read_csv(submission)

        for _, serial in submission_dataframe.iterrows():
            name = serial['id']
            value = list(map(int, serial['encoding'].split(' ')))
            value = [x * weight for x in value]

            if name not in rst.keys():
                rst[name] = [0 for i in range(len(value))]

            rst[name] = list(map(lambda x, y: x + y, rst[name], value))

    for name in rst.keys():
        value = rst[name]
        value = list(map(lambda x: str(int(x)), value))
        rst[name] = ' '.join(value)

    writer = open(out_submission_file, mode='w')
    writer.write('id,encoding\n')
    for k, v in rst.items():
        writer.write(k + ',' + v + '\n')
    writer.close()


            

    
