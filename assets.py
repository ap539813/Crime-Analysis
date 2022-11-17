logistic_regression = 'saved_models/LogisticRegression'
decision_tree = 'saved_models/DecisionTreeClassifier'
standerdscaler = 'saved_models/StandardScaler'
onehotencoder = 'saved_models/OneHotEncoder'

theme_image_name = 'assets_files/Crime_homepage.png'
column_names = ['Division', 'location_type', 'premises_type', 'ucr_code', 'ucr_ext',
       'offence', 'reportedyear', 'reportedmonth', 'reportedday',
       'reporteddayofyear', 'reporteddayofweek', 'reportedhour',
       'occurrenceyear', 'occurrencemonth', 'occurrenceday',
       'occurrencedayofyear', 'occurrencedayofweek', 'occurrencehour',
       'cleared', 'Neighbourhood', 'Longitude', 'Latitude']

months = ['January', 'February', 'March',
            'April', 'May', 'June', 
            'July', 'August', 'September', 
            'October', 'November', 'December']

mode_data = {'Division': 'D51',
            'location_type': 'Apartment (Rooming House, Condo)',
            'premises_type': 'Outside',
            'ucr_code': 1430.0,
            'ucr_ext': 100.0,
            'offence': 'Assault',
            'reportedyear': 2019.0,
            'reportedmonth': 'May',
            'reportedday': 18.0,
            'reporteddayofyear': 1.0,
            'reporteddayofweek': 'Monday',
            'reportedhour': 15.0,
            'occurrenceyear': 2019.0,
            'occurrencemonth': 'May',
            'occurrenceday': 1.0,
            'occurrencedayofyear': 1.0,
            'occurrencedayofweek': 'Friday',
            'occurrencehour': 0.0,
            'cleared': 'NO',
            'Neighbourhood': 'Waterfront Communities-The Island',
            'Longitude': -79.51578755,
            'Latitude': 43.61208553}


division = ['D31', 'D42', 'D22', 'D53', 'D51', 'D33', 'D14', 'D13', 'D11',
       'D12', 'D43', 'D32', 'D52', 'D54', 'D23', 'D55', 'D41', 'NSA']

location_type = ['Apartment (Rooming House, Condo)',
       'Single Home, House (Attach Garage, Cottage, Mobile)',
       'Open Areas (Lakes, Parks, Rivers)',
       'Other Commercial / Corporate Places (For Profit, Warehouse, Corp. Bldg',
       'Convenience Stores',
       'Parking Lots (Apt., Commercial Or Non-Commercial)', 'Unknown',
       'Commercial Dwelling Unit (Hotel, Motel, B & B, Short Term Rental)',
       'Streets, Roads, Highways (Bicycle Path, Private Road)',
       'Bar / Restaurant',
       "Other Non Commercial / Corporate Places (Non-Profit, Gov'T, Firehall)",
       'Group Homes (Non-Profit, Halfway House, Social Agency)',
       'Hospital / Institutions / Medical Facilities (Clinic, Dentist, Morgue)',
       'Schools During Un-Supervised Activity',
       'Police / Courts (Parole Board, Probation Office)',
       'Bank And Other Financial Institutions (Money Mart, Tsx)',
       'Jails / Detention Centres', 'Schools During Supervised Activity',
       'Construction Site (Warehouse, Trailer, Shed)',
       'Religious Facilities (Synagogue, Church, Convent, Mosque)',
       'Ttc Bus', 'Universities / Colleges', 'Ttc Subway Train',
       'Ttc Subway Station', 'Go Train', 'Retirement Home',
       'Gas Station (Self, Full, Attached Convenience)', 'Ttc Street Car',
       'Homeless Shelter / Mission',
       'Private Property Structure (Pool, Shed, Detached Garage)',
       'Dealership (Car, Motorcycle, Marine, Trailer, Etc.)',
       'Ttc Bus Stop / Shelter / Loop', 'Go Station',
       'Other Passenger Train', 'Other Regional Transit System Vehicle',
       'Other Passenger Train Station', 'Other Train Tracks',
       'Ttc Subway Tunnel / Outdoor Tracks',
       'Other Train Admin Or Support Facility', 'Go Bus',
       'Ttc Admin Or Support Facility', 'Ttc Light Rail Vehicle',
       'Cargo Train', 'Ttc Bus Garage', 'Ttc Light Rail Transit Station',
       'Other Train Yard', 'Ttc Wheel Trans Vehicle', 'Pharmacy',
       'Nursing Home', 'Halfway House', 'Community Group Home',
       'Ttc Support Vehicle']

premises_type = ['Apartment', 'House', 'Outside', 'Commercial', 'Other',
       'Educational', 'Transit']

ucr_code = [1430, 2120, 2130, 1610, 2132, 1460, 1420, 2135, 1480, 1457, 1410,
       1450, 1470, 1461, 1440, 2133, 1455, 2121, 1462, 1475, 2125, 1611]

ucr_ext = [100, 200, 210, 220, 110, 180, 120, 140, 150, 130, 160, 230, 170,
       190, 211, 215]

offence = ['Assault', 'B&E', 'Theft Over', 'Robbery - Business',
       'Theft From Motor Vehicle Over', "B&E W'Intent",
       'Assault - Force/Thrt/Impede', 'Assault Peace Officer',
       'Assault With Weapon', 'Theft Of Motor Vehicle',
       'Assault - Resist/ Prevent Seiz', 'Robbery - Other',
       'Pointing A Firearm', 'Robbery With Weapon', 'Aggravated Assault',
       'Assault Bodily Harm', 'Robbery - Mugging',
       'Robbery - Financial Institute', 'Unlawfully In Dwelling-House',
       'Robbery - Swarming', 'Discharge Firearm With Intent',
       'Robbery - Vehicle Jacking', 'Robbery - Purse Snatch',
       'Crim Negligence Bodily Harm', 'Discharge Firearm - Recklessly',
       'Theft From Mail / Bag / Key', 'Assault Peace Officer Wpn/Cbh',
       'Robbery - Home Invasion', 'Robbery - Taxi',
       'Unlawfully Causing Bodily Harm', 'Robbery - Armoured Car',
       'B&E Out', 'Theft Over - Shoplifting',
       'Use Firearm / Immit Commit Off', 'Robbery - Delivery Person',
       'Theft - Misapprop Funds Over', 'Robbery - Atm',
       'Aggravated Assault Avails Pros', 'Administering Noxious Thing',
       'B&E - To Steal Firearm', 'Aggravated Aslt Peace Officer',
       'Disarming Peace/Public Officer', 'Theft Of Utilities Over',
       'Air Gun Or Pistol: Bodily Harm', 'Theft Over - Distraction',
       'Traps Likely Cause Bodily Harm', 'Theft Over - Bicycle',
       'Set/Place Trap/Intend Death/Bh', 'B&E - M/Veh To Steal Firearm',
       'Robbery To Steal Firearm', 'Hoax Terrorism Causing Bodily']

neighbourhood = ['York University Heights', 'Malvern', 'Long Branch',
       'Thorncliffe Park', 'Islington-City Centre West',
       'North St.James Town', 'Pleasant View', 'Trinity-Bellwoods',
       'Humewood-Cedarvale', 'Roncesvalles', 'LAmoreaux',
       'Pelmo Park-Humberlea', 'Centennial Scarborough', 'Niagara',
       'Forest Hill North', 'Westminster-Branson', 'Moss Park',
       'Bay Street Corridor', 'Waterfront Communities-The Island',
       'Milliken', 'Banbury-Don Mills', 'Bendale',
       'Glenfield-Jane Heights', 'OConnor-Parkview', 'West Hill',
       'Weston', 'Kingsview Village-The Westway', 'Downsview-Roding-CFB',
       'Church-Yonge Corridor', 'Mount Olive-Silverstone-Jamestown',
       'Agincourt South-Malvern West',
       'Dovercourt-Wallace Emerson-Junction', 'Black Creek',
       'Yorkdale-Glen Park', 'Humbermede', 'Bayview Village',
       'Bedford Park-Nortown', 'Eglinton East', 'Forest Hill South',
       'High Park North', 'Cliffcrest', 'Agincourt North',
       'Edenbridge-Humber Valley', 'Mimico', 'Beechborough-Greenbrook',
       'Highland Creek', 'Oakwood Village', 'Mount Pleasant West',
       'Newtonbrook East', 'Willowdale East', 'University',
       'Broadview North', 'Little Portugal', 'St.Andrew-Windfields',
       'Casa Loma', 'Annex', 'North Riverdale', 'Englemount-Lawrence',
       'Humber Summit', 'Hillcrest Village', 'Tam OShanter-Sullivan',
       'Palmerston-Little Italy', 'Regent Park', 'Woburn',
       'West Humber-Clairville', 'Kingsway South', 'Dorset Park',
       'Stonegate-Queensway', 'Victoria Village', 'Lambton Baby Point',
       'Clairlea-Birchmount', 'Parkwoods-Donalda',
       'Bridle Path-Sunnybrook-York Mills', 'South Parkdale',
       'Weston-Pellam Park', 'Bathurst Manor', 'Taylor-Massey', 'Ionview',
       'Flemingdon Park', 'Keelesdale-Eglinton West',
       'Leaside-Bennington', 'Oakridge', 'Alderwood', 'Maple Leaf',
       'South Riverdale', 'Danforth', 'Willowridge-Martingrove-Richview',
       'Woodbine-Lumsden', 'Kensington-Chinatown', 'Wexford/Maryvale',
       'Birchcliffe-Cliffside', 'Caledonia-Fairbank',
       'Brookhaven-Amesbury', 'Steeles', 'Don Valley Village',
       'High Park-Swansea', 'Rouge', 'Bayview Woods-Steeles',
       'Rosedale-Moore Park', 'Cabbagetown-South St.James Town',
       'Kennedy Park', 'Greenwood-Coxwell', 'New Toronto',
       'Danforth East York', 'The Beaches', 'Briar Hill-Belgravia',
       'Playter Estates-Danforth', 'Corso Italia-Davenport',
       'Dufferin Grove', 'Etobicoke West Mall', 'Rustic',
       'Rockcliffe-Smythe', 'NSA', 'Runnymede-Bloor West Village',
       'East End-Danforth', 'Lawrence Park North', 'Blake-Jones',
       'Newtonbrook West', 'Wychwood', 'Willowdale West',
       'Eringate-Centennial-West Deane', 'Old East York', 'Mount Dennis',
       'Yonge-Eglinton', 'Morningside', 'Woodbine Corridor',
       'Scarborough Village', 'Junction Area', 'Princess-Rosethorn',
       'Clanton Park', 'Humber Heights-Westmount', 'Guildwood',
       'Lawrence Park South', 'Henry Farm', 'Markland Wood',
       'Yonge-St.Clair', 'Lansing-Westgate', 'Elms-Old Rexdale',
       'Thistletown-Beaumond Heights', 'Mount Pleasant East',
       'Rexdale-Kipling']


css_file_path = 'style/style.css'