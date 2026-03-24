const EXAMPLE_DIRECTORY = 'example';

const EXAMPLE_FILE_TYPES: Record<string, string> = {
  '.hdf': 'application/octet-stream',
  '.stl': 'model/stl',
  '.tif': 'image/tiff',
  '.vrt': 'application/xml',
};

function exampleAssetUrl(filename: string): string {
  return `${import.meta.env.BASE_URL}${EXAMPLE_DIRECTORY}/${filename}`;
}

function contentTypeFor(filename: string): string {
  const lowerName = filename.toLowerCase();
  const matchedExtension = Object.keys(EXAMPLE_FILE_TYPES).find((extension) => lowerName.endsWith(extension));
  return matchedExtension ? EXAMPLE_FILE_TYPES[matchedExtension] : 'application/octet-stream';
}

export async function fetchExampleFile(filename: string): Promise<File> {
  const response = await fetch(exampleAssetUrl(filename));
  if (!response.ok) {
    throw new Error(`The example asset could not be loaded: ${filename}`);
  }

  const bytes = await response.arrayBuffer();
  return new File([bytes], filename, { type: contentTypeFor(filename) });
}

export async function fetchExampleTerrainFiles(): Promise<File[]> {
  return Promise.all([
    fetchExampleFile('example.hdf'),
    fetchExampleFile('example.tif'),
    fetchExampleFile('example.vrt'),
  ]);
}

export async function fetchExampleStlFile(): Promise<File> {
  return fetchExampleFile('example.stl');
}
