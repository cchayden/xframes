FROM ipython/scipyserver

MAINTAINER XFrames Project <cchayden@gmail.com>

VOLUME /notebooks
WORKDIR /base

EXPOSE 8888
EXPOSE 80

#xFrames Pre-reqs
RUN apt-get update --fix-missing
RUN apt-get install -y wget default-jdk emacs24 nginx

ENV SPARK_HOME /usr/local/spark
ENV SPARK_VER spark-2.1.0
ENV PY4J_VER py4j-0.10.4-src.zip

RUN wget -q http://apache.cs.utah.edu/spark/spark-2.1.0/spark-2.1.0-bin-hadoop2.7.tgz -O /tmp/${SPARK_VER}.tgz && \
    tar -zxvf /tmp/${SPARK_VER}.tgz --directory /opt && ln -s /opt/${SPARK_VER}-bin-hadoop2.7 ${SPARK_HOME}

#Build Docs
RUN pip install --upgrade pip && pip install --upgrade sphinx && easy_install cloud-sptheme
RUN pip install mock numpydoc
#RUN mkdir /notebooks/notebooks

# Setting Environment Variables
ENV USE_HTTP 1
ENV LIB=/base
ENV PYTHONPATH="${SPARK_HOME}/python/:/base:/usr/local/spark/python/lib/${PY4J_VER}"
ENV SPARK_CONF_DIR=/base/conf

ADD docs /base/docs
ADD server.sh /
ADD xframes /base/xframes
ADD xframes/conf /base/conf
#ADD intro /notebooks/notebooks/intro

CMD ["/server.sh"]
