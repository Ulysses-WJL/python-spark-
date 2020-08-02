'''
@Author: ulysses
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-08-02 20:11:11
@LastEditors: Please set LastEditors
@Description: 
'''
import socket

server = socket.socket()

server.bind(('localhost', 9999))

server.listen(1)

while 1:

    print("I'm waiting the connect...")

    conn,addr = server.accept()  # 阻塞 等待连接
    
    print("Connect success! Connection is from %s " % addr[0])
   
    print('Sending data...')
    conn.send('I love hadoop I love spark hadoop is good spark is fast'.encode())
    conn.close()
    # 关闭连接, while 循环
    print('Connection is broken.')
