export interface Author {
  name: string;
  url?: string;
  affiliations: number[];
  isEqualContribution?: boolean;
  isCorresponding?: boolean;
}

export interface Affiliation {
  id: number;
  name: string;
}

export interface LinkButton {
  label: string;
  iconClass: string;
  url: string;
  isImage?: boolean;
}

export interface CarouselItem {
  id: string | number;
  title?: string;
  videoUrl?: string; // URL for mp4
  posterUrl?: string;
  youtubeId?: string; // Or YouTube ID
  description?: string;
}