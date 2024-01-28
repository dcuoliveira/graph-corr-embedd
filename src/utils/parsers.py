
def str_2_bool(val):

    val = str(val)

    if val.lower() == "false":
        return False
    elif val.lower() == "true": 
        return True
    else:
        raise Exception("Invalid boolean value: {}".format(val))