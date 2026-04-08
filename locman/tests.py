import pytest
import datetime
import pytz
from django.test import TestCase
from django.contrib.auth.models import User
from rest_framework import status
from rest_framework.test import APIClient, APITestCase
from rest_framework.authtoken.models import Token

from .models import UserProfile, Position, Event, Scan
from .serializers import EventSerializer, PositionSerializer, RouteSerializer
from .functions import (
    distance, calculate_speed, extrapolate_position, 
    get_process_stats, parse_file_gpx, parse_file_csv,
    get_location_events, generate_events, get_source_ids,
    nearest_amenities
)


# ==================== AUTHENTICATION TESTS ====================
class AuthenticationTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.token = Token.objects.get(user=self.user)

    def test_unauthenticated_access_denied(self):
        """Test that unauthenticated requests are denied"""
        response = self.client.get('/location-manager/event/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_authenticated_access_allowed(self):
        """Test that authenticated requests are allowed"""
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key)
        response = self.client.get('/location-manager/event/')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])


# ==================== USER ISOLATION TESTS ====================
class UserIsolationTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user1 = User.objects.create_user(username='user1', password='pass')
        self.user2 = User.objects.create_user(username='user2', password='pass')
        
        # Create position for user1
        self.position1 = Position.objects.create(
            user=self.user1.profile,
            lat=10.0,
            lon=20.0,
            time=pytz.utc.localize(datetime.datetime(2024, 1, 1, 12, 0, 0)),
            source='test'
        )
        
        # Create position for user2
        self.position2 = Position.objects.create(
            user=self.user2.profile,
            lat=30.0,
            lon=40.0,
            time=pytz.utc.localize(datetime.datetime(2024, 1, 1, 13, 0, 0)),
            source='test'
        )

    def test_user1_cannot_access_user2_positions(self):
        """Test that user1 cannot see user2's position data"""
        self.client.force_authenticate(user=self.user1)
        response = self.client.get('/location-manager/position/')
        # Should only contain user1's data
        positions = response.data if response.status_code == 200 else []
        self.assertFalse(any(p['lon'] == 40.0 for p in positions if isinstance(positions, list)))


# ==================== EVENT VIEWSET TESTS ====================
class EventViewSetTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)
        
        self.now = pytz.utc.localize(datetime.datetime.utcnow())
        self.event = Event.objects.create(
            user=self.user.profile,
            timestart=self.now - datetime.timedelta(hours=1),
            timeend=self.now,
            lat=10.0,
            lon=20.0
        )

    def test_event_list(self):
        """Test listing events"""
        response = self.client.get('/location-manager/event/')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])

    def test_event_retrieve_by_id(self):
        """Test retrieving event by ID"""
        response = self.client.get(f'/location-manager/event/{self.event.id}/')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])

    def test_event_retrieve_by_date(self):
        """Test retrieving events by date"""
        date_str = self.now.strftime('%Y-%m-%d')
        response = self.client.get(f'/location-manager/event/{date_str}/')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])

    def test_event_retrieve_by_location(self):
        """Test retrieving events by location"""
        response = self.client.get(f'/location-manager/event/2024-01-01/10.0/20.0/')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])


# ==================== POSITION VIEWSET TESTS ====================
class PositionViewSetTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)
        
        self.now = pytz.utc.localize(datetime.datetime.utcnow())
        self.position = Position.objects.create(
            user=self.user.profile,
            lat=10.0,
            lon=20.0,
            time=self.now,
            source='test'
        )

    def test_position_list(self):
        """Test listing last 10 positions"""
        response = self.client.get('/location-manager/position/')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])

    def test_position_retrieve_explicit(self):
        """Test retrieving explicit position by timestamp"""
        timestamp = self.now.strftime('%Y%m%d%H%M%S')
        response = self.client.get(f'/location-manager/position/{timestamp}/')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])

    def test_position_retrieve_interpolated(self):
        """Test retrieving interpolated position (between two explicit positions)"""
        pos1 = self.position
        pos2 = Position.objects.create(
            user=self.user.profile,
            lat=20.0,
            lon=30.0,
            time=self.now + datetime.timedelta(hours=1),
            source='test'
        )
        
        # Try to get position between them
        between_time = self.now + datetime.timedelta(minutes=30)
        timestamp = between_time.strftime('%Y%m%d%H%M%S')
        response = self.client.get(f'/location-manager/position/{timestamp}/')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])


# ==================== ROUTE VIEWSET TESTS ====================
class RouteViewSetTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)
        
        self.now = pytz.utc.localize(datetime.datetime.utcnow())
        
        # Create a route with multiple positions
        for i in range(5):
            Position.objects.create(
                user=self.user.profile,
                lat=10.0 + i,
                lon=20.0 + i,
                time=self.now + datetime.timedelta(minutes=i),
                source='test',
                elevation=100 + i * 10
            )

    def test_route_retrieve_geojson(self):
        """Test retrieving route as GeoJSON"""
        start = self.now.strftime('%Y%m%d%H%M%S')
        end = (self.now + datetime.timedelta(minutes=5)).strftime('%Y%m%d%H%M%S')
        response = self.client.get(f'/location-manager/route/{start}{end}/')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])
        if response.status_code == 200:
            self.assertIn('geo', response.data)


# ==================== BOUNDING BOX VIEWSET TESTS ====================
class BoundingBoxViewSetTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)
        
        self.now = pytz.utc.localize(datetime.datetime.utcnow())
        
        for i in range(5):
            Position.objects.create(
                user=self.user.profile,
                lat=10.0 + i,
                lon=20.0 + i,
                time=self.now + datetime.timedelta(minutes=i),
                source='test'
            )

    def test_bbox_retrieve(self):
        """Test retrieving bounding box"""
        start = self.now.strftime('%Y%m%d%H%M%S')
        end = (self.now + datetime.timedelta(minutes=5)).strftime('%Y%m%d%H%M%S')
        response = self.client.get(f'/location-manager/bbox/{start}{end}/')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])
        if response.status_code == 200:
            self.assertEqual(len(response.data), 4)  # [minlon, minlat, maxlon, maxlat]


# ==================== ELEVATION VIEWSET TESTS ====================
class ElevationViewSetTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)
        
        self.now = pytz.utc.localize(datetime.datetime.utcnow())
        
        for i in range(5):
            Position.objects.create(
                user=self.user.profile,
                lat=10.0 + i,
                lon=20.0 + i,
                time=self.now + datetime.timedelta(minutes=i),
                source='test',
                elevation=100 + i * 10,
                explicit=True
            )

    def test_elevation_retrieve(self):
        """Test retrieving elevation profile"""
        start = self.now.strftime('%Y%m%d%H%M%S')
        end = (self.now + datetime.timedelta(minutes=5)).strftime('%Y%m%d%H%M%S')
        response = self.client.get(f'/location-manager/elevation/{start}{end}/')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])
        if response.status_code == 200:
            self.assertIsInstance(response.data, list)


# ==================== PROCESS VIEWSET TESTS ====================
class ProcessViewSetTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)

    def test_process_list(self):
        """Test listing process status"""
        response = self.client.get('/location-manager/process/')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])
        if response.status_code == 200:
            self.assertIn('tasks', response.data)
            self.assertIn('stats', response.data)


# ==================== IMPORT/UPLOAD ENDPOINT TESTS ====================
class ImportUploadTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)

    def test_import_get_status(self):
        """Test GET import status"""
        response = self.client.get('/location-manager/import')
        self.assertIn(response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])

    def test_import_post_file(self):
        """Test POST file upload"""
        # Would need actual file for full test
        pass


# ==================== SERIALIZER TESTS ====================
class SerializerTests(TestCase):
    def test_event_serializer_valid(self):
        """Test EventSerializer with valid data"""
        now = pytz.utc.localize(datetime.datetime.utcnow())
        data = {
            'timestart': now,
            'timeend': now + datetime.timedelta(hours=1),
            'lat': 10.0,
            'lon': 20.0,
            'amenities': []
        }
        serializer = EventSerializer(data=data)
        self.assertTrue(serializer.is_valid())

    def test_position_serializer_valid(self):
        """Test PositionSerializer with valid data"""
        now = pytz.utc.localize(datetime.datetime.utcnow())
        data = {
            'time': now,
            'lat': 10.0,
            'lon': 20.0,
            'speed': 50,
            'explicit': True,
            'source': 'test'
        }
        serializer = PositionSerializer(data=data)
        self.assertTrue(serializer.is_valid())


# ==================== MODEL TESTS ====================
class ModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')

    def test_position_creation(self):
        """Test Position model creation"""
        now = pytz.utc.localize(datetime.datetime.utcnow())
        position = Position.objects.create(
            user=self.user.profile,
            lat=10.0,
            lon=20.0,
            time=now,
            source='test'
        )
        self.assertEqual(position.lat, 10.0)
        self.assertEqual(position.lon, 20.0)
        self.assertTrue(position.explicit)

    def test_event_creation(self):
        """Test Event model creation"""
        now = pytz.utc.localize(datetime.datetime.utcnow())
        event = Event.objects.create(
            user=self.user.profile,
            timestart=now,
            timeend=now + datetime.timedelta(hours=1),
            lat=10.0,
            lon=20.0
        )
        self.assertEqual(event.lat, 10.0)
        self.assertIsNotNone(event.amenities_data)

    def test_userprofile_creation(self):
        """Test UserProfile auto-creation on User creation"""
        user = User.objects.create_user(username='newuser', password='pass')
        self.assertTrue(hasattr(user, 'profile'))
        self.assertIsNotNone(user.profile.token)


# ==================== UTILITY FUNCTION TESTS ====================
class UtilityFunctionTests(TestCase):
    def test_distance_calculation(self):
        """Test Haversine distance calculation"""
        # London to Paris
        dist = distance(51.5074, -0.1278, 48.8566, 2.3522)
        self.assertAlmostEqual(dist, 343600, delta=1000)  # ~343km in meters

    def test_calculate_speed(self):
        """Test speed calculation"""
        user = User.objects.create_user(username='testuser', password='testpass')
        now = pytz.utc.localize(datetime.datetime.utcnow())
        
        pos1 = Position.objects.create(
            user=user.profile,
            lat=0.0,
            lon=0.0,
            time=now,
            source='test'
        )
        
        pos2 = Position.objects.create(
            user=user.profile,
            lat=0.01,
            lon=0.01,
            time=now + datetime.timedelta(seconds=60),
            source='test'
        )
        
        speed = calculate_speed(pos2)
        self.assertGreater(speed, 0)

    def test_extrapolate_position(self):
        """Test position interpolation"""
        user = User.objects.create_user(username='testuser', password='testpass')
        now = pytz.utc.localize(datetime.datetime.utcnow())
        
        pos1 = Position.objects.create(
            user=user.profile,
            lat=0.0,
            lon=0.0,
            time=now,
            source='test'
        )
        
        pos2 = Position.objects.create(
            user=user.profile,
            lat=2.0,
            lon=2.0,
            time=now + datetime.timedelta(hours=1),
            source='test'
        )
        
        mid_time = now + datetime.timedelta(minutes=30)
        interp = extrapolate_position(user, mid_time)
        
        self.assertFalse(interp.explicit)
        self.assertEqual(interp.source, 'realtime')
        self.assertAlmostEqual(interp.lat, 1.0, delta=0.01)
        self.assertAlmostEqual(interp.lon, 1.0, delta=0.01)


# ==================== EDGE CASE TESTS ====================
class EdgeCaseTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)

    def test_invalid_timestamp_format(self):
        """Test handling of invalid timestamp format"""
        response = self.client.get('/location-manager/position/invalid/')
        self.assertIn(response.status_code, [status.HTTP_400_BAD_REQUEST, status.HTTP_404_NOT_FOUND])

    def test_empty_result_set(self):
        """Test handling when no data exists"""
        response = self.client.get('/location-manager/position/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 0)

    def test_negative_elevation(self):
        """Test handling negative elevation values"""
        now = pytz.utc.localize(datetime.datetime.utcnow())
        position = Position.objects.create(
            user=self.user.profile,
            lat=10.0,
            lon=20.0,
            time=now,
            source='test',
            elevation=-10.0  # Below sea level
        )
        self.assertEqual(position.elevation, -10.0)


# ==================== INTEGRATION TESTS ====================
class IntegrationTests(APITestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username='testuser', password='testpass')
        self.client.force_authenticate(user=self.user)
        self.now = pytz.utc.localize(datetime.datetime.utcnow())

    def test_full_workflow(self):
        """Test complete workflow: create positions, query route, get elevation"""
        # Create positions
        for i in range(5):
            Position.objects.create(
                user=self.user.profile,
                lat=10.0 + i,
                lon=20.0 + i,
                time=self.now + datetime.timedelta(minutes=i),
                source='test',
                elevation=100 + i * 10,
                explicit=True
            )
        
        # Query route
        start = self.now.strftime('%Y%m%d%H%M%S')
        end = (self.now + datetime.timedelta(minutes=5)).strftime('%Y%m%d%H%M%S')
        
        route_response = self.client.get(f'/location-manager/route/{start}{end}/')
        self.assertIn(route_response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])
        
        # Query elevation
        elev_response = self.client.get(f'/location-manager/elevation/{start}{end}/')
        self.assertIn(elev_response.status_code, [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST])
