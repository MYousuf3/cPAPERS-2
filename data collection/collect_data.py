import openreview
import os
import subprocess
import arxiv
import json
from tqdm import tqdm
import time

def retrieve_submissions(client, invitation):
    submissions = client.get_all_notes(invitation=invitation, details='replies')
    return submissions

def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def open_json(path):
    with open(path, 'r') as file:
        return json.load(file)

def get_arxiv_id(client, title):
    try:
        search = arxiv.Search(
            query = f"ti:{title}",
            max_results = 1)
        
        if next(client.results(search)) is not None:
            url = str(next(client.results(search)))
            id = url.split('/')[-1]
            return id
        
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Could not find paper_id for {title}")
        return None

def download_paper(client, paper_id, path):
    paper = next(client.results(arxiv.Search(id_list=[paper_id])))
    paper.download_source(filename=f"{path}tar_files/{paper_id}.tar.gz")

def collect_reviews(client, submissions, path, download_papers=False):
    grabbed_submissions = open_json(path + 'raw_qas.json')
    ids_already_grabbed = set(submission['submission_id'] for submission in grabbed_submissions)
    print(f"Already grabbed {len(ids_already_grabbed)} submissions")
    previous_title = None
    previous_paper_id = None
    for submission in tqdm(submissions):
        if submission.id in ids_already_grabbed:
            continue

        try:
            for reply in submission.details['replies']:
                if reply['invitation'].endswith('Official_Review'):

                    submission_id = submission.id
                    title = submission.content['title']
                    if title == previous_title:
                        paper_id = previous_paper_id
                    else:
                        print(f"Collecting paper_id for {title}")
                        paper_id = get_arxiv_id(client, title)
                        if paper_id is None:
                            print(f"Could not find paper_id for {title}")
                            continue

                        previous_paper_id = paper_id
                        previous_title = title

                    print("Sleeping for 1 second...")
                    time.sleep(1)
                    
                    if paper_id is not None:
                        if download_papers and not os.path.exists(path + f"tar_files/{paper_id}.tar.gz"):
                            print(f"Downloading paper {paper_id}")
                            download_paper(client, paper_id, path)

                        # print(submission)
                        # print(reply['content'])
                        review_info = {
                            'submission_id': submission_id,
                            'paper_id': paper_id,
                            'review_id': reply['id'],
                            'submission_title': title,
                            # had to change from review to main_review and to questions to strength_and_weaknesses
                            'review': reply['content']['main_review'],
                        }
                    else:
                        continue

                    grabbed_submissions.append(review_info)

            save_json(grabbed_submissions, path + 'raw_qas.json')
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

    return "Reviews collected!"

def collect_comments(submissions, submissions_data, path):
    for submission in tqdm(submissions):
        try:
            for reply in submission.details['replies']:
                if reply['invitation'].endswith('Official_Comment'):            
                    for review in submissions_data:
                        if review['review_id'] == reply['replyto']:
                            review['comment'] = reply['content']['comment']
                        else:
                            continue
                    save_json(submissions_data, path + 'raw_qas.json')
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

    return "Comments collected!"

def unpack_tar_files(path):
    for file in tqdm(os.listdir(os.path.join(path, 'tar_files'))):
        file_path = os.path.join(path, 'tar_files', file)
        extract_path = os.path.join(path + 'extracted_files', os.path.splitext(file)[0])
        os.makedirs(extract_path, exist_ok=True)
        
        # Determine if the file is a gzip-compressed tar archive
        if file.endswith('.tar.gz') or file.endswith('.tgz'):
            command = ['tar', '-xzf', file_path, '-C', extract_path]
        elif file.endswith('.tar'):
            command = ['tar', '-xf', file_path, '-C', extract_path]
        else:
            print(f"Unsupported file format: {file}")
            continue
        
        result = subprocess.run(command, capture_output=True)
        
        if result.returncode != 0:
            # If there's an error, print it out
            print(f"Error extracting {file}: {result.stderr.decode('utf-8')}")


def main():
    client = arxiv.Client()
    guest_client = openreview.Client(baseurl='https://api.openreview.net')

    
    conference = 'NeurIPS.cc/2021/Conference/-/Blind_Submission'
    path = './neurips/2021/'
    
    # conference = 'NeurIPS.cc/2022/Conference/-/Blind_Submission'
    # path = './neurips/2022/'

    # conference = 'NeurIPS.cc/2023/Conference/-/Blind_Submission'
    # path = './neurips/2023/'
    
    # conference = 'ICLR.cc/2020/Conference/-/Blind_Submission'
    # path = './iclr/2020/'

    # venues = guest_client.get_group(id='venues').members
    # print([i for i in venues if 'ICML.cc/2022/' in i])

    # for year in [2020,2021,2022,2023]:
    #     submissions = guest_client.get_all_notes(invitation=f"NeurIPS.cc/{year}/Conference/-/Blind_Submission", details='directReplies')
    #     print(f"NeurIPS-{year}: {len(submissions)}")

    # for year in [2020,2021,2022,2023]:
    #     submissions = guest_client.get_all_notes(invitation=f"ICLR.cc/{year}/Conference/-/Blind_Submission", details='directReplies')
    #     print(f"ICLR-{year}: {len(submissions)}")

    # for year in [2020,2021,2022,2023]:
    #     submissions = guest_client.get_all_notes(invitation=f"ICML.cc/{year}/Conference/-/Blind_Submission", details='directReplies')
    #     print(f"ICML-{year}: {len(submissions)}")

    unpack = True
    download_papers = True

    # Ensure the directory exists
    file_name = 'raw_qas.json'
    file_path = os.path.join(path, file_name)
    os.makedirs(path, exist_ok=True)
    save_json([], file_path)

    # Retrieve submissions from OpenReview Client
    submissions = retrieve_submissions(guest_client, conference)
    print(f"Retrieved {len(submissions)} submissions")
    
    collect_reviews(client, submissions, path, download_papers)

    submissions_data = open_json(path + 'raw_qas.json')
    print(len(submissions_data))
    collect_comments(submissions, submissions_data, path)

    if unpack:
        unpack_tar_files(path)

if __name__ == '__main__':
    main()