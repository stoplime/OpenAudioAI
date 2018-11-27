import resource
def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return "{}: usertime={} systime={} mem={} mb".format(
            point, 
            format(round(usage[0], 6), '.6f'), 
            format(round(usage[1], 4), '.4f'),
            format(round((usage[2]*resource.getpagesize())/1000000.0, 3), '.3f') )

def main():
    print(using("Label"))

if __name__ == "__main__":
    main()