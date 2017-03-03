FROM neuralensemble/simulation

USER docker
RUN /home/docker/env/neurosci/bin/pip3 install --upgrade pip
RUN /home/docker/env/neurosci/bin/pip3 install jupyter
RUN /home/docker/env/neurosci/bin/pip3 install pandas
RUN /home/docker/env/neurosci/bin/pip3 install seaborn

USER root

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8888
CMD ["/home/docker/env/neurosci/bin/jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
