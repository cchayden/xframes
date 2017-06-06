FROM jupyter/pyspark-notebook

MAINTAINER XFrames Project <cchayden@gmail.com>

VOLUME /notebooks

EXPOSE 8888
EXPOSE 4040-4048

USER root
RUN apt-get update --fix-missing
RUN apt-get install -y emacs24
USER $NB_USER

COPY xframes /base/xframes
COPY xframes/conf /base/conf
COPY server.sh /usr/local/bin

# Setting Environment Variables
ENV LIB=/base
ENV SPARK_CONF_DIR=/base/conf
ENV XFRAMES_CONFIG_DIR=/base/conf

CMD server.sh