import json
import argparse

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

table_schema = {
    'fields': [
        {'name': 'author_id', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'thread_id', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'created_time', 'type': 'TIMESTAMP', 'mode': 'NULLABLE'},
        {'name': 'username', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'name', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'message', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'thread_target', 'type': 'STRING', 'mode': 'NULLABLE'},
        {'name': 'sentiment_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
        {'name': 'sentiment_magnitude', 'type': 'FLOAT', 'mode': 'NULLABLE'},
    ]
}

class Flatten(beam.DoFn):
    def process(self, item):
        if item.get('data'):
            data = item.get('data')
            includes = item.get('includes')
            rules = item.get('matching_rules')
            
            row = {
                'author_id': data.get('author_id'),
                'thread_id': data.get('id'),
                'created_time': data.get('created_at'),
                'username': includes.get('users')[0].get('username'),
                'name': includes.get('users')[0].get('name'),
                'message': data.get('text'),
                'thread_target': rules[0].get('tag')
            }

            yield row

class NLP(beam.DoFn):
    def process(self, item):

        try:
            from google.cloud import language
            from google.cloud.language import enums
            from google.cloud.language import types

            client = language.LanguageServiceClient()
            
            text = item.get('message')
            
            document = language.types.Document(
                content=text,
                type=language.enums.Document.Type.PLAIN_TEXT)

            sentiment = client.analyze_sentiment(document=document).document_sentiment

            item['sentiment_score'] = sentiment.score
            item['sentiment_magnitude'] = sentiment.magnitude

            yield item
        except Exception as err:
            print(err)
            return

def run(argv=None):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--topic',
        dest='topic')
    parser.add_argument(
        '--temp_location',
        dest='temp')
    parser.add_argument(
        '--table_spec',
        dest='table_spec',
        help='Bigquery table path. Ex: dataset.table')
    parser.add_argument(
        '--project',
        dest='project',
        help='Google Cloud Platform project ID')

    known_args, pipeline_args = parser.parse_known_args(argv)

    pipeline_options = PipelineOptions(
        argv=pipeline_args,
        streaming=True,
        runner='DataflowRunner',
        project=known_args.project,
        temp_location=known_args.temp,
        region='us-central1',
        requirements_file='./requirements.txt'
    )

    with beam.Pipeline(options=pipeline_options) as pipeline:
        (
            pipeline
            | "PubSub messages" >> beam.io.ReadFromPubSub(topic=f'projects/{known_args.project}/topics/{known_args.topic}')
            | "Parse PubSub" >> beam.Map(lambda payload: json.loads(payload.decode('utf-8')))
            | "Flatten payload" >> beam.ParDo(Flatten())
            | "Sentiment Discover" >> beam.ParDo(NLP())
            | "Writing on Bigquery" >> beam.io.WriteToBigQuery(
                    table=f'{known_args.project}:{known_args.table_spec}', 
                    schema=table_schema,
                    write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND, 
                    create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                    custom_gcs_temp_location=known_args.temp
                )
        )

if __name__ == '__main__':
    run()