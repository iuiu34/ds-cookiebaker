"""Data drift."""
import datetime as dt
import os

import fire
from edo.mkt import bq
from slack_sdk import WebClient


def post_message_to_slack(text='Hello world!'):
    """Post message to slack."""
    client = WebClient(token=os.environ['SLACK_NOTIFICATION_OAUTH_TOKEN'])
    response = client.chat_postMessage(
        channel='#svc-ds-mkt-ltv-py',
        text=text,
        icon_emoji=":palm_tree:")
    assert response["ok"]


def get_sql_avg(metric, table, start_date, end_date,
                partition_column='booking_date'):
    """Get sql avg."""
    sql = f"""
        select avg({metric}) {metric}
        from `{table.bq}`
        where date({partition_column}) between date('{start_date}') and date('{end_date}') - 1
    """
    data = bq.get_query(sql)
    if len(data) != 1:
        raise ValueError(f"{len(data)=}")
    return data[metric].values[0]


def data_drift(tablename: str = None,
               start_date: str = None,
               end_date: str = None,

               project: str = 'ds-mkt',
               dataset: str = 'ds_ltv',
               slack: bool = True,
               partition_column: str = 'booking_date',
               end_date_days: int = 14):
    """Data drift."""
    print('data drift')
    print(vars())
    table = bq.TableName(project, dataset, tablename)
    # AVG_THRESHOLD: float = 0.05
    AVG_THRESHOLD: float = 0
    metrics = ['ltv_issued_future_18']
    for metric in metrics:
        metric_avg = get_sql_avg(metric, table, start_date, end_date, partition_column)
        start_date_ = dt.date.fromisoformat(start_date)
        start_date_ -= dt.timedelta(days=end_date_days)
        end_date_ = dt.date.fromisoformat(end_date)
        end_date_ -= dt.timedelta(days=end_date_days)

        metric_avg_ = get_sql_avg(metric, table, start_date_, end_date_, partition_column)
        metric_avg_ratio = metric_avg_ / metric_avg - 1
        if abs(metric_avg_ratio) > AVG_THRESHOLD:
            text = f"`{metric}` has a {metric_avg_ratio:.0%} (>10%) " \
                   f"increase WoW (34.2 € vs 29.2€)\n" \
                   f"Scope: all markets"
            print(text)
            # text = f"{metric} has a discrepancy of {metric_avg_ratio:.0%}"
            if slack:
                emoji = ":large_orange_circle:"
                text_ = 3 * emoji + '\n' + text + '\n' + 3 * emoji
                post_message_to_slack(text_)
            # raise ValueError(text.replace('\n', '. '))


def main():
    """Execute main program."""
    fire.Fire(data_drift)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
