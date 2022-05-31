# Simple API Using Opencv with AI model

## To Start Services 
```
sudo cp ai_test_server.service /etc/systemd/system/
sudo systemctl start ai_test_server
sudo systemctl enable ai_test_server
```
## To Check service is runing 
```
sudo systemctl status ai_test_server
```


## Tutorial

> Before we setup nginx,we need to setup gunicorn such that it can be started by systemd. This is required for nginx.

1.  Gunicorn by systemd
```
sudo vi /etc/systemd/system/myapp.service
```
Add following lines to it:
```
[Unit]
Description=Gunicorn instance to serve My flask app
After=network.target
[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/my_code
Environment="PATH=/home/ubuntu/anaconda3/bin"
ExecStart=/home/ubuntu/anaconda3/envs/myapp/bin/gunicorn --workers 3 --bind unix:myapp.sock -m 007 connector:app
[Install]
WantedBy=multi-user.target
```
To enable linking:

```
sudo systemctl start myapp
sudo systemctl enable myapp
sudo systemctl status myapp
```
2. Nginx setup
```
sudo apt-get install nginx php7.4-fpm php7.4 php7.4-json
sudo usermod ubuntu -g www-data
```
create nginx configuration file:
```
sudo vi /etc/nginx/sites-available/myapp
server {
 listen 80;
 server_name x.x.x.x;
location / {
 include proxy_params;
 proxy_pass http://unix:/home/ubuntu/my_code/myapp.sock;
 }
}
```
now,create symlink
```
sudo ln -s /etc/nginx/sites-available/myapp /etc/nginx/sites-enabled
```
test application:
```
sudo systemctl start nginx
```
>Application will be running at 'http://localhost:5000'

3. Monitoring and logs

To enable your application logging, add following code into flask app code:
```
import logging
logging.basicConfig(filename=APP_ROOT+'/'+'execution_log.log', filemode='a+', format=' [%(filename)s:%(lineno)s:%(funcName)s()]- %(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
 
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)
```
commands to start and stop server:

Gunicorn:
```
ps -ef|grep gunicorn    
sudo systemctl enable myapp
sudo systemctl status myapp
sudo systemctl start myapp
sudo systemctl stop myapp
```
Nginx:
```
ps -ef | grep nginx
sudo systemctl status nginx.service
sudo systemctl start nginx 
sudo systemctl stop nginx 
sudo systemctl restart nginx
sudo less /var/log/nginx/error.log: checks the Nginx error logs.
sudo less /var/log/nginx/access.log: checks the Nginx access logs.
sudo journalctl -u nginx: checks the Nginx process logs.
sudo journalctl -u myapp: checks your Flask app’s Gunicorn logs.
```

# Setup NGINX with PHP support to run PHP code
>https://www.pcsuggest.com/setup-nginx-with-php/