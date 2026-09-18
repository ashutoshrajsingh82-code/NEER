# NEER — Frontend Dockerfile
# Neural Embedding based Estimation and Reconstruction (SIH26066, MoES/INCOIS)

FROM node:20-slim AS base
WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
