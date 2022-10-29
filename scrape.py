import multiprocessing as mp
import os

import requests
import tqdm
from bs4 import BeautifulSoup

PAGE_URL = 'https://pokemondb.net/pokedex/national'
PAGE_PREFIX = 'https://pokemondb.net'


def get_soup_from_link(link: str, add_prefix=False) -> BeautifulSoup:
    """
    :param link: a url to request
    :param add_prefix: whether to add PAGE_PREFIX in the beginning or not
    :return: a BeautifulSoup object of the requested url
    """
    page_url = PAGE_PREFIX + link if add_prefix else link
    page_html = requests.get(page_url).text
    return BeautifulSoup(page_html, 'lxml')


def card_info(card) -> {'id': int, 'name': str, 'link': str}:
    """
    A function to return necessary information from a pokemon card
    :param card: A card of a pokemon in the URL
    :return: JSON-like dict with id (pokedex number), pokemon name and a link for more info
    """

    link = card.a['href']
    info = card.find('span', class_='infocard-lg-data text-muted')
    id = int(info.small.text[1:])
    name = info.find(class_='ent-name').text
    return {
        'id': id,
        'name': name,
        'link': link
    }


def process_artwork_link(artwork_link) -> [str]:
    """
    :param artwork_link: a suffix of the location of the pokemon's additional artwork link
    :return: a list of all images' urls in that page.
    """
    soup = get_soup_from_link(artwork_link, add_prefix=True)
    images = soup.find_all('img')[1:]
    images_links = [image['src'] for image in images]

    lazy_images = soup.find_all('a', class_='lazyload-resp-inner')
    lazy_links = [lazy_image['href'] for lazy_image in lazy_images]
    return images_links + lazy_links


def download_images_to_path(links: [str], save_path: str) -> None:
    """
    :param links: A list of link of images to download
    :param save_path: a location to save the images in
    :return: the function doesn't return anything
    """
    for i, link in enumerate(links):
        name = link.split('/')[-1]
        name = name[:-4] + str(i) + name[-4:]
        file_name = os.path.join(save_path, name)
        r = requests.get(link)
        with open(file_name, 'wb') as outfile:
            outfile.write(r.content)


def process_individual_pokemon(card) -> None:
    """
    :param card: Pokemon's card
    :return: None (Only downloads all images to the correct place)
    """
    card = BeautifulSoup(card, 'lxml')
    info = card_info(card)
    cover_image_link = card.find(class_="img-fixed img-sprite")['data-src']
    soup = get_soup_from_link(info['link'], add_prefix=True)
    boxes = soup.find_all('a', class_='sprite-share-link')
    sprites_links = [box['href'] for box in boxes]
    additional_artwork_link = soup.find('a', string='Additional artwork')
    images_links = [cover_image_link]
    if additional_artwork_link is not None:
        additional_artwork_link = additional_artwork_link['href']
        images_links = images_links + process_artwork_link(additional_artwork_link)

    dir_name = f"{'{0:03}'.format(info['id'])}. {info['name']}"
    save_path = f"./data/" + dir_name
    save_path_sprites = save_path + '/sprites'
    if not os.path.exists(save_path_sprites):
        os.makedirs(save_path_sprites)
    download_images_to_path(images_links, save_path)
    download_images_to_path(sprites_links, save_path_sprites)


def process_pokemon(pokemon) -> None:
    try:
        process_individual_pokemon(pokemon)
    except NotADirectoryError:
        pass


def get_all_images():
    """
    This function downloads all images from https://pokemondb.net/pokedex/national
    and saves them to ./data folder, with separate folder for each Pokémon.
    """
    html_text = requests.get(PAGE_URL).text
    soup = BeautifulSoup(html_text, 'lxml')

    # Converted to string and then back to BeautifulSoup to bypass recursion max size problem
    # Solution taken from here - https://stackoverflow.com/a/52021597
    pokemons_info = [str(card) for card in soup.find_all('div', class_='infocard')]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(process_pokemon, pokemons_info), total=len(pokemons_info)):
            pass
