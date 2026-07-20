FROM nginx:alpine

# Copy index.html static client directly into Nginx html root
COPY index.html /usr/share/nginx/html/index.html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
