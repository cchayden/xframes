FROM jupyter/pyspark-notebook

MAINTAINER XFrames Project <cchayden@gmail.com>

VOLUME /notebooks

EXPOSE 8888
EXPOSE 4040-4048

USER root
RUN apt-get update --fix-missing
RUN apt-get install -y emacs24
USER $NB_USER

RUN pip2 install xframes

# Setting Environment Variables
ENV LIB=/base
ENV SPARK_CONF_DIR=/base/conf
ENV XFRAMES_CONFIG_DIR=/base/conf

COPY conf /base/conf

CMD ["start-notebook.sh", "--NotebookApp.token=''"]