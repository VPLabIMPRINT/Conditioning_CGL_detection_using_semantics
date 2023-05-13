import CGL_test_server as cgl_server
from multiprocessing import Process, Queue
import time

# def queuer(q):
#     while True:
#         q.put("JOB")
#         print "Adding JOB"
#         time.sleep(1)

# def worker(q):  
#     while True:
#         if not q.empty():
#             item = q.get()
#             print "Running", item
#         else:
#             print "No jobs"
#             time.sleep(1)



if __name__ == '__main__':
    print("fd")
    CGL_q = Queue()
    # Nav_q = Queue()
    a = Process(target=cgl_server.initialize, args=(CGL_q,))
    # b = Process(target=, args=(Nav_q,))
    a.start()
    # b.start()

    CGL_q.put("something added")
    time.sleep(10)
    CGL_q.put("something added again")
    time.sleep(10)
    CGL_q.put("something added again")
    time.sleep(10)
    CGL_q.put("something added again")
