import re
import os
from mtcnn import MTCNN

# validate the username inputted
def validate_username(username):
    if not username.strip():
        print("username cannot be empty")
        return False
    
    if len(username) < 3 or len(username) > 20:
        print("username must be between 2 and 30 characters.")
        return False

    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        print("⚠️ Only letters, numbers, and underscores are allowed.")
        return False

    if username.lower() in ['admin', 'system', 'root']:
        print("⚠️ This username is reserved. Choose another.")
        return False
    return True


# list of registered people
DIR = r'/home/aayushi/fv2/dataset'
registered = [i for i in os.listdir(DIR)]

# model function
mt = MTCNN()