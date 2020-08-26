#!/usr/bin/python
# -*- coding: utf-8 -*-

import pika
import time
import json
import sys
import os
import nltk
import time
sys.path.append('.')

from processing import *

# connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
# channel = connection.channel()
#
# channel.queue_declare(queue='hello')
#
# def callback(ch, method, properties, body):
#     print(" [x] Received %r" % body)
#
#
# channel.basic_consume(callback,
#                           queue='hello',
#                           no_ack=True)
#
#
# print(' [*] Waiting for messages. To exit press CTRL+C')
# channel.start_consuming()


credentials = pika.PlainCredentials('admin_loom', '')

connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmqdocker', 5672, '/', credentials))

channel = connection.channel()

channel.queue_declare(queue='rpc_queue')

#Download NLTK dictionary etc.
nltk.download('punkt')
#initialize_dictionaries('punkt')

# load predictive models

#print(os.listdir('./opt/python_scripts'))
#print(os.listdir('./opt/python_scripts/models'))

#Load the models
try:
    fvecs, models = loadModels('./opt/python_scripts/models')
except:
    print("Can not load the models\n")
    e = sys.exc_info()
    print(e)


####################################
### definition error message #######
####################################

output_error_de = {"ops":[{"insert":"Leider ist ein Fehler aufgetreten, bitte versuchen Sie es später nocheinmal"},
                          {"attributes":{"bold":True},"insert":"8"},
                         ]}

output_error_en = {"ops":[{"insert":"An error occured, please try again"},
                          {"attributes":{"bold":True},"insert":"8"},
                         ]}


###################################
####   Method Def #################
###################################


def on_request(ch, method, props, body):

    #calculat time
    start_time = time.time()

    # Here the body is being decoded from bytes array to string
    try:
        json_as_string = body.decode()
    except:
        print("\nError cannot decode body to string\n")
        e = sys.exc_info()
        print(e)

    # final_object is a JSON parsed from json_as_string
    try:
        final_object = json.loads(json_as_string)
    except:
        print("\nJSON string cannot be loaded\n")
        e = sys.exc_info()
        print(e)

    try:
        # get the value of reviewId from finalobject
        f = int(final_object['assignedFileId'])
    except:
        print("\nReview file ID can not be loaded\n")
        e = sys.exc_info()
        print(e)

    # checking if reviewId has been fetched properly
    print(final_object)

    # creation of new response
    try:
        response = {'assignedFileId':    final_object['assignedFileId']}
    except:
        print("Error in creating a response object")
        e = sys.exc_info()
        print(e)

    # calculations
    feedback = final_object['formattedReviewText']

    # Error handling
    try:
        response.update(processFeedback(feedback, fvecs, models))
    except:
        # Central log entry
        print("\n Error in processFeedback Method\n")
        e = sys.exc_info()
        print(e)
        # send respsonse that it was not possible to calculate the quality
        response.update({'output_error_en': json.dumps(output_error_en),
                         'output_error_de': json.dumps(output_error_de) })

    # Transform json to string (json)
    json_response = json.dumps(response)

    # publish the response
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id = \
                                                         props.correlation_id),
                     body=str(json_response))
    ch.basic_ack(delivery_tag = method.delivery_tag)

    # Duration of calculation
    end_time = time.time()
    elapsed = end_time - start_time
    print("\n\n time calc: ", elapsed, "\n\n")

# Max consumers at once (for now please set 1 as below)
channel.basic_qos(prefetch_count=1)

channel.basic_consume(on_request, queue='rpc_queue')

print(" [x] Awaiting RPC requests")

try:
    channel.start_consuming()
except:
    print("Error in consuming method")
    e = sys.exc_info()
    print(e)