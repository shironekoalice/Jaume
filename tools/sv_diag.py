import faulthandler, sys
faulthandler.dump_traceback_later(8, exit=True)
print("starting import")
import funasr.models.sense_voice.model as m
print("ok")