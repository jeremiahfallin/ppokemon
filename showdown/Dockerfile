FROM node:18-slim

RUN apt-get update && apt-get install -y git procps && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth=1 https://github.com/smogon/pokemon-showdown.git
WORKDIR /opt/pokemon-showdown
RUN npm install --production

# Optional: copy custom config
# COPY ./config/config.js ./config/config.js
# RUN cp config/config-example.js config/config.js

EXPOSE 8000
CMD ["node", "pokemon-showdown", "start", "--no-security", "--port", "8000"]