from distutils.log import error
import logging, datetime,os
from functools import wraps
from pathlib import Path
from PythIon.utils import Singleton
import inspect

# default_log_level=20 #INFO


try:
    user_dir=Path.home()
    PYTHION_APPDATA_DIR=os.path.join(user_dir,"PythionData")
    LOG_BACKUP_DIR=os.path.join(PYTHION_APPDATA_DIR,"backupLogs")
    LOG_BACKUP_FNAME_DEFAULT="backup.log"
    if not os.path.exists(PYTHION_APPDATA_DIR):
        os.mkdir(PYTHION_APPDATA_DIR)
    if not os.path.exists(LOG_BACKUP_DIR):
        os.mkdir(LOG_BACKUP_DIR)
except Exception as e:
    print("initialization error -- failed to create appdata directories\n",e)
    




class LogSystem(metaclass=Singleton):
    def __init__(self,stream=True,backup=True,filename="",
                 backupFilename=os.path.join(LOG_BACKUP_DIR,LOG_BACKUP_FNAME_DEFAULT)):
        self.backup=backup
        self.stream=stream
        self.logfile=filename
        self.backupFilename=backupFilename
        
    def set_logfile(self,filename):
        self.logfile=filename
        
    def log(self,modulename,excludeparams=[],isMethod=True,preprocessor=None,postprocessor=None,lograwresult=False,**logkwargs):
        
        def decorate(fn):
            sig = inspect.signature(fn)
            @wraps(fn)
            def wrapper(*args,**kwargs):
                # logstr="="*10+'\n'
                # logstr+=datetime.now().
                # print(fn, fn.__name__)z
                # print(dir(fn))
                now=datetime.datetime.now()
                nowstr="@"+now.strftime("%Y-%m-%d, %H:%M:%S.%f %Z |")+str(now.timestamp())
                self._dumpln(nowstr)
                
                fnstr="!"+modulename
                self._dumpln(fnstr)
                
                result = fn(*args,**kwargs)
                if lograwresult:
                    self._dump("^Returned:"+str(result))
                if preprocessor !=None:
                    try:
                        prestr=getattr(args[0],preprocessor)(*args,**kwargs)
                        self._dumpln(f"*Arguments: {prestr}")
                    except Exception as e:
                        print("logging error: pre-processor failed")
                        print(e)
                else:
                    bound = sig.bind(*args, **kwargs) # this takes care of args and default kwargs issues
                    bound.apply_defaults()
                    try:
                        bound.arguments.pop("self")
                    except:
                        pass
                    for param in excludeparams: #remove parameters that are specified to be excluded
                        bound.arguments.pop(param)
                    bargs=bound.arguments
                    self._dumpln("*Arguments: "+str(bargs))
                    
                    
                if postprocessor !=None:
                    try:
                        poststr=getattr(args[0],postprocessor)()
                        self._dumpln(poststr)
                    except Exception as e:
                        print("logging error: post-processor failed")
                        print(e)
                
                
                return result
            return wrapper
        return decorate
    
    def _dump(self, text):
        if self.stream: 
            print(text)
        if self.backup:
            with open(self.backupFilename, "a") as backupfile:
                backupfile.write(text)
        if self.logfile!="":
            try:
                with open(self.logfile,"a") as logfile:
                    logfile.write(text)
            except Exception as e:
                print("Logging error:", e)
    
    def _dumpln(self, text):
        self._dump(text+"\n")
        
                
            
            
    
# logger=LogSystem(backup=True,filename=os.path.join(PYTHION_APPDATA_DIR,"test.log"))

# @logger.log("name")
# def dummy (a):
#     print("in dummy: ", a)
    
    
# class Dummy():
#     def __init__(self):
#         self.a=1
#         pass
    
#     @logger.log("Dummy.parse",postprocessor="post")
#     def parse(self,abc=True):
#         self.a=2
#         return [1,2]
    
#     def post(self):
#         return "Result:" +str(self.a*2)

# d=Dummy()
# d.parse(False)
# print(LogSystem)


        

    

