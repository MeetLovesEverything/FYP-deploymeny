{ pkgs }: {
  deps = [
    pkgs.libsndfile
    pkgs.libvorbis
    pkgs.ffmpeg
    pkgs.python310Full
  ];
}