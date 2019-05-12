const express = require('express');
var app = express();
var server = require('http').createServer(app);
var io = require('socket.io')(server);
var vtb_current_ip = '';

app.get('/vtuber', (req, res) => res.sendFile(__dirname + '/vtuber.html'));
app.use(express.static('public'));

io.of('/sakuya').on('connection', (socket) => {
    console.log('a cds client connected');

    socket.on('change_ip', (msg) => {
        vtb_current_ip = msg;
        console.log(msg);
    });

    socket.on('frame_data', (msg) => {
        socket.emit('response', vtb_current_ip);
        if (msg != 0) {
            socket.broadcast.emit('frame_download', msg.toString('base64'));
        }
    });

    socket.on('result_data', (result) => {
        socket.broadcast.emit('result_download', result);
    });

    socket.on('disconnect', () => { console.log('a cds client disconnected') });
});

server.listen(6789, () => console.log('listening on *:6789'));
