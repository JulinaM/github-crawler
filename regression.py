import sys
from utils import print_json
from utils.cloudant_utils import cloudant_db as db, save_doc
from datetime import date
import numpy as np
import pandas as pd


def main(args):
    fields = ["_id", "releases", "stars", "watchers", "commits", "forks", "issues"]
    repos = [r for r in db.get_query_result({
        "type": "release"
    }, ["_id", "releases"], limit=10000, raw_result=True)["docs"]]
    # columns = repos[0]['releases'][0].keys()
    print(repos[0]['releases'][0].keys())
    values = [r for release in repos for r in release["releases"]]
    df = pd.DataFrame(values)
    # df = df[df['release_date'] > '2020-01-01']
    # df['release_count'] = df.groupby('repo')['release_tag'].transform('count')
    # df['release_count'] = df.groupby('repo')['release_tag'].transform('count')


    # print(repos.groups)
    # for name, group in repos:
    # print(name, 'contains', group.shape[0], 'releases')
    # print(df)
    # print(df.get_group('10up/classifai').count())
    new_df = df.groupby('repo').agg(
        forks=pd.NamedAgg(column="forks", aggfunc="sum"),
        downloads=pd.NamedAgg(column="downloads", aggfunc="sum"),
        issues=pd.NamedAgg(column="issues", aggfunc="sum"),
        stars=pd.NamedAgg(column="stars", aggfunc="sum"),
        commits=pd.NamedAgg(column="commits", aggfunc="sum"),
        release_counts=pd.NamedAgg(column="release_tag", aggfunc="count")
    )
    print(new_df.shape)
    print(list(new_df.columns))

    print(new_df)


if __name__ == "__main__":
    main(sys.argv[1:])
