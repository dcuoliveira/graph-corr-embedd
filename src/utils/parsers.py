
def str_2_list(val):
    return [int(x) for x in val.split(",")]

def str_2_bool(val):

    val = str(val)

    if val.lower() == "false":
        return False
    elif val.lower() == "true": 
        return True
    else:
        raise Exception("Invalid boolean value: {}".format(val))