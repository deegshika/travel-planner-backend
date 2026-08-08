import os
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


def build_chroma():
    load_dotenv()
    CHROMA_DIR = os.getenv("CHROMA_DIR", "./.chromadb")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    # Create embedding function using OpenAI
    ef = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-3-small")

    # Persistent Chroma client
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Create or get collection
    collection = client.get_or_create_collection(name="travel_guides", embedding_function=ef)

    # Seed data: list of dicts with id, city, text (including rough costs in INR)
    items = [
        {"id": "1", "city": "Bangalore", "text": "Bangalore Palace: historic Tudor-style palace with ornate interiors and grounds. Entry ~INR 230 for Indians, INR 460 for foreigners; camera fee extra (~INR 200)."},
        {"id": "2", "city": "Bangalore", "text": "Lalbagh Botanical Garden: famous glasshouse and centuries-old trees. Entry ~INR 20; flower shows may have small fees. Great for photography at sunrise."},
        {"id": "3", "city": "Bangalore", "text": "Cubbon Park: large green space in the city center with colonial-era buildings nearby. Free entry; ideal for casual street and nature photography."},
        {"id": "4", "city": "Bangalore", "text": "MG Road & Brigade Road food stalls and cafes: try filter coffee and local South Indian meals. Budget ~INR 100-400 per person depending on venue."},
        {"id": "5", "city": "Bangalore", "text": "VV Puram Food Street: popular evening street-food destination offering dosas, chaats, and sweets. Expect ~INR 50-300 per dish."},
        {"id": "6", "city": "Bangalore", "text": "Nandi Hills (near Bangalore): sunrise viewpoint and photography spot ~60–90 km away. Petrol/toll and entry combined ~INR 300-800 depending on transport; minimal entry fee ~INR 15-50."},
        {"id": "7", "city": "Bangalore", "text": "Commercial Street: shopping and colorful street scenes; good for candid photography. Small purchases and snacks ~INR 200-1500 depending on shopping."},
        {"id": "8", "city": "Bangalore", "text": "Toit / Arbor Brewing Company: popular microbreweries in Indiranagar for food and craft beer. Meal + drinks ~INR 800-2000 per person depending on consumption."},
        {"id": "9", "city": "Bangalore", "text": "Tipu Sultan's Summer Palace: timber structure and historical exhibits. Entry ~INR 15-100; photography allowed for small fee."},
        {"id": "10", "city": "Bangalore", "text": "Ulsoor Lake & waterfront: boating and sunrise photo opportunities. Boat rides ~INR 200-500 depending on duration and group."},
        {"id": "11", "city": "Kodaikanal", "text": "Kodaikanal Lake: pedal boating and cycling around the star-shaped lake. Boat rides and bike rental ~INR 80-250 depending on duration."},
        {"id": "12", "city": "Kodaikanal", "text": "Coaker's Walk: scenic paved path on the ridge offering valley views and sunrise photography. Free entry; small parking or guide fees may apply (~INR 20-50)."},
        {"id": "13", "city": "Kodaikanal", "text": "Bryant Park: botanical garden with flowers, ferns, and glasshouses. Entry ~INR 30; ideal for plant photography and relaxed strolls."},
        {"id": "59", "city": "Kodaikanal", "text": "Devonshire Restaurant: iconic hilltop eatery known for hot chocolate, bakery items, and Anglo-Indian meals. Tea/meal combos ~INR 250-450 per person depending on choices."},
        {"id": "60", "city": "Kodaikanal", "text": "Muncheez Cafe: popular vegetarian spot for dosas, sandwiches, and homemade juices. Budget ~INR 120-280 per person."},
        {"id": "61", "city": "Kodaikanal", "text": "Dolphin's Nose viewpoint: dramatic rock ledge with panoramic valley views and photo-worthy cliffs. Small entry/parking fee ~INR 50-100."},
        {"id": "62", "city": "Kodaikanal", "text": "Pillar Rocks: three giant rock columns rising above the valley with misty views. Entry ~INR 40-80; great for landscape photography."},
        {"id": "63", "city": "Kodaikanal", "text": "Silver Cascade Falls: roadside waterfall formed by outflow from Kodaikanal Lake. Free to visit; small parking or tea stall costs ~INR 20-60."},
        {"id": "64", "city": "Kodaikanal", "text": "Shembaganur Museum of Natural History: museum and orchidarium with local plants, wildlife exhibits, and photographic displays. Entry ~INR 50-100."},
        {"id": "65", "city": "Kodaikanal", "text": "Kurinji Andavar Temple: hilltop temple dedicated to Lord Murugan with sweeping views of the valley and seasonal kurinji blooms. Free entry; modest donation ~INR 20-50."},
        {"id": "66", "city": "Kodaikanal", "text": "Bear Shola Falls: peaceful forest waterfall with a short walk from the road. Free access; small local guide or parking charge ~INR 20-50."},
        {"id": "67", "city": "Kodaikanal", "text": "Kodaikanal Pancakes or hillside bakery stop: try local pancakes, tea, and grilled sandwiches after a day of sightseeing. Snack prices ~INR 80-250 per person."},
        {"id": "68", "city": "Kodaikanal", "text": "A lesser-known local gem: Vattakanal village trails and quiet viewpoints beyond the main lake district. Ideal for offbeat photography and local tea stalls ~INR 50-150 for snacks and transport."},
        {"id": "69", "city": "Ooty", "text": "Ooty Lake boat rides: pedal boats and rowboats on a scenic lake surrounded by eucalyptus trees and hills. Boat rental ~INR 150-300 per person, plus snacks at lakeside stalls."},
        {"id": "70", "city": "Ooty", "text": "Doddabetta Peak: highest point in the Nilgiris with panoramic views and photography opportunities. Entry ~INR 30; binoculars and tea at summit ~INR 50-100."},
        {"id": "71", "city": "Ooty", "text": "Botanical Gardens: 55-acre garden with exotic plants, roses, and fern house. Entry ~INR 30 for Indians, INR 100 for foreigners; ideal for nature photography."},
        {"id": "72", "city": "Ooty", "text": "Taj Savoy Hotel high tea or lunch: historic colonial hotel offering traditional Nilgiri cuisine and elegant dining. Meal ~INR 600-1200 per person."},
        {"id": "73", "city": "Ooty", "text": "Tea estate visit in nearby Coonoor: guided tour of tea gardens, factory demonstration, and tasting. Tour cost ~INR 200-500 plus transport ~INR 150-300."},
        {"id": "74", "city": "Ooty", "text": "Rose Garden at the railway station: small rose garden and colorful flower displays near the Ooty Toy Train station. Free entry; snacks and chai nearby ~INR 40-100."},
        {"id": "75", "city": "Ooty", "text": "Pine Forest: iconic foggy pine grove on the way to Ooty Lake, popular for photography and peaceful short walks. Free to visit; local guide/parking charge ~INR 20-50."},
        {"id": "76", "city": "Ooty", "text": "Charring Cross market food stalls: taste fresh varkey, homemade chocolates, and mountain herbal tea. Snacks ~INR 50-200 depending on items."},
        {"id": "77", "city": "Ooty", "text": "St. Stephen's Church: 19th-century Anglican church with stained glass and colonial architecture. Free entry; donations appreciated ~INR 20-50."},
        {"id": "78", "city": "Ooty", "text": "A lesser-known local gem: Avalanche Lake trail and trout farm beyond Ooty, offering quiet nature walks and serene views. Transport and entry ~INR 150-300; trout meals extra ~INR 200-400."},
        {"id": "14", "city": "Goa", "text": "Anjuna Beach flea market: browse clothing, jewelry, and souvenirs in the late afternoon. Snacks and small purchases ~INR 100-800 depending on items."},
        {"id": "15", "city": "Goa", "text": "Calangute Beach shacks: enjoy seafood, drink coconut water, and watch the sunset. Meal and drinks ~INR 300-1200 per person depending on choice."},
        {"id": "16", "city": "Goa", "text": "Fort Aguada: Portuguese fort with lighthouse and panoramic views of the Arabian Sea. Entry and photography ~INR 30-100; local transport extra."},
        {"id": "17", "city": "Jaipur", "text": "Hawa Mahal: iconic pink sandstone palace with honeycomb windows. Entry ~INR 200 for Indians, INR 500 for foreigners; photography allowed."},
        {"id": "18", "city": "Jaipur", "text": "Lassiwala on MI Road: try the city's famous sweet lassi and local snacks. Drink and light snacks ~INR 80-200 per person."},
        {"id": "19", "city": "Mumbai", "text": "Gateway of India: waterfront landmark with iconic arches and harbor views. Ferry rides and snacks nearby ~INR 150-300; best at sunrise or sunset for photography."},
        {"id": "20", "city": "Mumbai", "text": "Chor Bazaar: bustling antique market in South Mumbai with vintage finds and street scenes. Small purchases and street food ~INR 100-500 depending on items."},
        {"id": "21", "city": "Mumbai", "text": "Marine Drive promenade: famous curved coastline for evening strolls and cityscape photos. Free entry; local snacks and chai ~INR 50-150."},
        {"id": "22", "city": "Mumbai", "text": "Kala Ghoda cafe crawl: art district cafes with specialty coffee and desserts. Coffee and pastries ~INR 200-500 per person depending on venue."},
        {"id": "23", "city": "Mumbai", "text": "Sanjay Gandhi National Park: green oasis with hiking trails, Kanheri Caves, and birdwatching. Entry ~INR 60-120; guide/transport extra for caves ~INR 100-250."},
        {"id": "24", "city": "Mumbai", "text": "Carter Road and Bandra Bandstand: seaside walkways with colorful murals, sunset views, and casual eateries. Snacks and tea ~INR 100-250 per person."},
        {"id": "25", "city": "Mumbai", "text": "Juhu Beach street food: iconic beach with bhelpuri, pav bhaji, and local treats. Street food tasting ~INR 150-400 depending on items."},
        {"id": "26", "city": "Mumbai", "text": "Dhobi Ghat: open-air laundry with rhythmic lines of clothes and unique documentary photography opportunities. Free to view from the gallery; small donation or tea ~INR 50-100 if guided."},
        {"id": "27", "city": "Mumbai", "text": "Bandra Fort: historic seaside fort with Mumbai skyline views and sunset photography. Free entry; small transport cost ~INR 50-150."},
        {"id": "28", "city": "Mumbai", "text": "A lesser-known local gem: Khotachiwadi heritage village with quaint Portuguese-style houses and narrow lanes. Ideal for heritage photography; local tea stalls ~INR 40-120."},
        {"id": "29", "city": "Delhi", "text": "Humayun's Tomb: UNESCO World Heritage site with Mughal gardens and grand architecture. Entry ~INR 50 for Indians, INR 600 for foreigners; great for sunrise photography."},
        {"id": "30", "city": "Delhi", "text": "Chandni Chowk street food tour: sample parathas, chaat, jalebi, and kebabs in Old Delhi. Street food tasting ~INR 150-400 depending on dishes."},
        {"id": "31", "city": "Delhi", "text": "Lodhi Garden: peaceful park with historic tombs, walking trails, and photography-friendly greenery. Free entry; tea stalls nearby ~INR 30-80."},
        {"id": "32", "city": "Delhi", "text": "Qutub Minar complex: tall stone tower and ancient ruins with detailed carvings. Entry ~INR 40 for Indians, INR 650 for foreigners; excellent for architectural photos."},
        {"id": "33", "city": "Delhi", "text": "Dilli Haat: open-air craft bazaar and food village offering regional dishes and handicrafts. Entry ~INR 30; meal prices ~INR 150-450 per person."},
        {"id": "34", "city": "Delhi", "text": "Nizamuddin Dargah and Hazrat Nizamuddin Basti: historic Sufi shrine with lively qawwali evenings and charming alleyways. Free entry; small donations appreciated."},
        {"id": "35", "city": "Delhi", "text": "Hauz Khas Village: mix of medieval ruins, lake views, boutiques, and cafes. Entry to the ruins area free; cafe meals ~INR 250-600 per person."},
        {"id": "36", "city": "Delhi", "text": "Majnu ka Tilla: Tibetan market and lakeside neighborhood with colorful streets and budget eats. Ideal for candid travel photography; snacks ~INR 80-250."},
        {"id": "37", "city": "Delhi", "text": "India Gate and Rajpath: iconic war memorial and wide boulevard perfect for evening strolls and cityscape photos. Free entry; street vendors and chai ~INR 50-120."},
        {"id": "38", "city": "Delhi", "text": "A lesser-known local gem: Mehrauli Archaeological Park with quiet ruins, hidden stepwells, and peaceful trails. Free entry; perfect for offbeat photography."},
        {"id": "39", "city": "Goa", "text": "Arambol Beach: laid-back northern beach with sunrise views, hippie markets, and drum circles. Local shack meals ~INR 150-400; parasol rental ~INR 100-200."},
        {"id": "40", "city": "Goa", "text": "Old Goa churches: Basilica of Bom Jesus and Se Cathedral showcasing Baroque architecture and history. Entry free; guide tips ~INR 50-150; photography permitted in outdoor areas."},
        {"id": "41", "city": "Goa", "text": "Dudhsagar Waterfalls: dramatic four-tiered waterfall in the Western Ghats, popular for nature photography. Jeep/trek transport ~INR 400-800; small entry/guide fees ~INR 50-100."},
        {"id": "42", "city": "Goa", "text": "Ponda spice plantations: aromatic farm tours with traditional Goan meals and spice photography. Tour + lunch ~INR 500-900 per person."},
        {"id": "43", "city": "Goa", "text": "Palolem Beach: scenic south Goa bay with colorful boats and sunset photo spots. Beach huts and meals ~INR 250-700; boat ride ~INR 300-500."},
        {"id": "44", "city": "Goa", "text": "Titos Lane, Baga: nightlife strip with clubs, bars, and street food. Drinks and snacks ~INR 300-1200 depending on venue."},
        {"id": "45", "city": "Goa", "text": "Chapora Fort: hilltop ruins overlooking Vagator Beach, excellent for dramatic seaside photography. Free entry; small transport cost ~INR 80-200."},
        {"id": "46", "city": "Goa", "text": "Cotigao Wildlife Sanctuary: forest trails, waterfalls, and birdwatching in south Goa. Entry ~INR 30-100; guide/permit extra ~INR 150-300."},
        {"id": "47", "city": "Goa", "text": "Martin's Corner: popular seafood restaurant in Betalbatim known for Goan fish curry and prawns. Meal ~INR 500-1200 per person depending on seafood choices."},
        {"id": "48", "city": "Goa", "text": "A lesser-known local gem: Saligao village street art and quiet chapel lanes near Calangute. Ideal for cultural photography; tea or snack ~INR 50-150 at local cafes."},
        {"id": "49", "city": "Jaipur", "text": "Amer Fort: hilltop fort with Sheesh Mahal mirrors and sprawling courtyards. Entry ~INR 100-200 for Indians, INR 550-650 for foreigners; photography and light show extra."},
        {"id": "50", "city": "Jaipur", "text": "Jal Mahal: palace in the middle of Man Sagar Lake with sunrise reflections and palace views from the shore. Free to view from waterfront; small boat ride or camera fee ~INR 50-150."},
        {"id": "51", "city": "Jaipur", "text": "Nahargarh Fort and sunset point: panoramic city views popular with photographers. Entry ~INR 20-50; food at the fort cafe ~INR 150-350."},
        {"id": "52", "city": "Jaipur", "text": "Chokhi Dhani: cultural village experience with Rajasthani food, folk performances, and craft demonstrations. Buffet and entertainment ~INR 700-1200 per person."},
        {"id": "53", "city": "Jaipur", "text": "Galta Ji (Monkey Temple): hilltop temple complex with natural springs and photogenic water tanks. Entry free; modest prayer or parking donation ~INR 20-50."},
        {"id": "54", "city": "Jaipur", "text": "Panna Meena ka Kund: stepwell with geometric architecture ideal for travel photography. Free entry; local tea stall snacks ~INR 30-80."},
        {"id": "55", "city": "Jaipur", "text": "Laxmi Misthan Bhandar (LMB): famous sweets and traditional Rajasthani thali in Johri Bazaar. Meal and desserts ~INR 150-350 per person."},
        {"id": "56", "city": "Jaipur", "text": "Birla Mandir and Moti Dungri Hills: white marble temple with sunset views and calm gardens. Free entry; offerings and parking ~INR 20-50."},
        {"id": "57", "city": "Jaipur", "text": "Jaipur Wax Museum and heritage procession walk: unique cultural stop near Raj Mandir with colorful sculptures and photo ops. Entry ~INR 200-300."},
        {"id": "58", "city": "Jaipur", "text": "A lesser-known local gem: Bagru printing villages outside Jaipur, where craftsmen hand-block fabrics in traditional designs. Workshop visit and snacks ~INR 150-300."},
        {"id": "79", "city": "Mysore", "text": "Mysore Palace light show: illuminated royal palace and evening photo opportunity. Entry to palace ~INR 70 for Indians, INR 200 for foreigners; light show adds small fee ~INR 20-50."},
        {"id": "80", "city": "Mysore", "text": "Brindavan Gardens: terraced gardens with musical fountain and evening lighting. Entry ~INR 50-100; boat rides and snacks extra ~INR 100-250."},
        {"id": "81", "city": "Mysore", "text": "Mysore Masala Dosa at Vinayaka Mylari: iconic local eatery known for butter dosa and traditional South Indian breakfast. Meal ~INR 80-150 per person."},
        {"id": "82", "city": "Mysore", "text": "Chamundi Hill viewpoint and temple: hilltop temple with panoramic city views and sunrise photography. Temple donation/parking ~INR 30-70."},
        {"id": "83", "city": "Mysore", "text": "Sri Ranganathaswamy Temple, Srirangapatna: historic temple complex near Mysore with intricate carvings and calm riverside setting. Entry free; guide/donation ~INR 20-50."},
        {"id": "84", "city": "Mysore", "text": "Kerehalli Lake: quiet lakeside spot for birdwatching and rustic photography. Free to visit; small snack/tea stall costs ~INR 30-80."},
        {"id": "85", "city": "Mysore", "text": "Nadabrahma Satsang hall: classical music and dance performance space in a traditional Mysore building. Event ticket ~INR 150-400 depending on performance."},
        {"id": "86", "city": "Mysore", "text": "Food Street near Mysore Zoo: street vendors offering local chaats, sweets, and kebabs. Snacks ~INR 50-250 depending on dishes."},
        {"id": "87", "city": "Mysore", "text": "Mysore Zoo (Sri Chamarajendra Zoological Gardens): one of India’s oldest zoos with well-kept animals and landscaped enclosures. Entry ~INR 80 for Indians, INR 200 for foreigners; camera fee extra ~INR 50."},
        {"id": "88", "city": "Mysore", "text": "A lesser-known local gem: Devaraja Market back lanes, where traditional flower, spice, and sandalwood stalls create vivid street photography scenes. Local buys ~INR 50-200."},
        {"id": "89", "city": "Shimla", "text": "Mall Road cafes: cozy hilltop cafes serving tea, snacks, and momos with mountain views. Snack/tea combo ~INR 120-300 per person."},
        {"id": "90", "city": "Shimla", "text": "The Ridge and Christ Church: colonial-era church and open promenade popular for photography and seasonal fairs. Free entry; souvenirs and snacks ~INR 80-200."},
        {"id": "91", "city": "Shimla", "text": "Kufri adventure park: hillside fun park with zorbing, horse rides, and snow activities in winter. Activity packages ~INR 200-500."},
        {"id": "92", "city": "Shimla", "text": "Jakhoo Hill and Hanuman Temple: highest viewpoint in Shimla with panoramic town views and temple visit. Free temple entry; ropeway/parking ~INR 50-150."},
        {"id": "93", "city": "Shimla", "text": "Scandal Point and evening street food: iconic junction with local vendors offering chaat, nuts, and hot tea. Snacks ~INR 80-250."},
        {"id": "94", "city": "Shimla", "text": "Kalka-Shimla toy train views: narrow-gauge railway route with lush hills and heritage tunnels ideal for scenic photography. Train fare ~INR 150-400 depending on class."},
        {"id": "95", "city": "Shimla", "text": "Green Valley picnic spot: quiet nature area near Kufri with rhododendron groves and meadow views. Free entry; picnic snacks ~INR 100-250."},
        {"id": "96", "city": "Shimla", "text": "Himalayan Bird Park: small aviary with native Himalayan birds and flowering trees near the Mall. Entry ~INR 50-120; good for bird and nature photography."},
        {"id": "97", "city": "Shimla", "text": "Cafe Sol and bakery at Chotta Shimla: popular food stop for homemade cakes, coffee, and rustic sandwiches. Meal ~INR 150-350 per person."},
        {"id": "98", "city": "Simla", "text": "A lesser-known local gem: Tara Devi Temple trek route with forest trails, prayer flags, and quiet hilltop views. Free to visit; transport and chai ~INR 50-120."},
    ]

    # Prepare lists for Chroma
    ids = [item["id"] for item in items]
    documents = [item["text"] for item in items]
    metadatas = [{"city": item["city"]} for item in items]

    # Add or update entries in the collection
    # If items with same ids exist, we'll delete them first to ensure idempotent seed
    try:
        existing_ids = collection.get(ids=ids)
        if existing_ids and existing_ids.get("ids"):
            collection.delete(ids=ids)
    except Exception:
        # ignore if get/delete not supported yet or empty
        pass

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # Persist to disk
    try:
        client.persist()
    except Exception:
        # some Chroma deployments persist automatically
        pass

    print(f"Seeded {len(items)} documents into collection 'travel_guides' at {CHROMA_DIR}")


if __name__ == "__main__":
    build_chroma()
