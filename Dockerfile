FROM pytorch/pytorch
COPY . /root/projects/bfnet
COPY ./runscript.sh /
WORKDIR "/root/projects/bfnet/bfnet"
RUN cd .. && pip install -e .
ENTRYPOINT ["/runscript.sh"]
CMD []
