import sqlite3
import os

DATABASE = 'database.db'

def main():
    db = sqlite3.connect(DATABASE)
    c = db.cursor()

    os.chdir('static/mirflickr')
    for file_name in os.listdir("."):
        if file_name.endswith(".jpg"):
            print file_name
            
            c.execute("SELECT * FROM images WHERE file_name = '%s'" % file_name)
            if c.fetchone() == None:
                c.execute("INSERT INTO images (file_name) VALUES ('%s')" % file_name)
            else:
                continue
                
            c.execute("SELECT * FROM images WHERE file_name = '%s'" % file_name)
            pid = c.fetchone()[0]

            file_number = file_name.replace('.jpg', '').replace('im', '')
            f = open('meta/tags/tags{}.txt'.format(file_number))
            for tag in f:
                c.execute("SELECT * FROM tags WHERE content = '%s'" % tag)
                result = c.fetchone()
                if result == None:
                    c.execute("INSERT INTO tags (content) VALUES ('%s')" % tag)
                c.execute("SELECT * FROM tags WHERE content = '%s'" % tag)
                tid = c.fetchone()[0]
                c.execute("INSERT INTO tags_images (tid, pid) VALUES (%d, %d)" % (tid, pid))
                
    db.commit()

if __name__ == '__main__':
    main()