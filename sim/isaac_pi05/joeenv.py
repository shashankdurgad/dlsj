import site, sys
J = "/home/joe-spark-2/pi05-isaac-so101"
site.addsitedir(J + "/venv/lib/python3.12/site-packages")
sys.path.insert(0, "/home/yan/pi05bench/extra"); sys.path += [J + "/runtime/src", J + "/studio/library/src"]
