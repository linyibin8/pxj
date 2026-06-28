server {
    server_name pxj.evowit.com;
    client_max_body_size 80m;

    location / {
        proxy_pass http://100.64.0.13:8038;
        proxy_http_version 1.1;
        proxy_request_buffering off;
        proxy_buffering off;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 240s;
        proxy_send_timeout 240s;
    }


    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/pxj.evowit.com/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/pxj.evowit.com/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot

}

server {
    if ($host = pxj.evowit.com) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


    server_name pxj.evowit.com;
    listen 80;
    return 404; # managed by Certbot


}
