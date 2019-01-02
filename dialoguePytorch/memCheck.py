import os

if os.name != 'nt':
    import resource

def using(point=""):
    if os.name == 'nt':
        return "{}: usertime={} systime={} mem={} mb".format("N/a", "N/a", "N/a", "N/a")
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