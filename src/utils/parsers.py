
def str_2_bool(str):
    if str.lower() == "false":
        return False
    elif str.lower() == "true": 
        return True
    else:
        raise Exception("Invalid boolean value: {}".format(str))