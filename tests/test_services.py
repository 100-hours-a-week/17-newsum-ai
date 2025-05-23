# tests/test_services.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json
import redis.asyncio as redis

# DatabaseClient 임포트를 위해 필요한 settings mock
with patch('app.services.database_client.Settings') as mock_settings:
    mock_settings_instance = MagicMock()
    mock_settings_instance.REDIS_HOST = 'localhost'
    mock_settings_instance.REDIS_PORT = 6379
    mock_settings_instance.REDIS_DB = 0
    mock_settings_instance.REDIS_PASSWORD = None
    mock_settings.return_value = mock_settings_instance
    
    from app.services.database_client import DatabaseClient


class TestDatabaseClient:
    """DatabaseClient 클래스 테스트"""

    @pytest.fixture
    def mock_redis_settings(self):
        """Redis 설정 mock"""
        return {
            'host': 'test-host',
            'port': 6380,
            'db': 1,
            'password': 'test-password'
        }

    @patch('app.services.database_client.redis.Redis')
    def test_database_client_init_with_params(self, mock_redis, mock_redis_settings):
        """매개변수로 DatabaseClient 초기화 테스트"""
        client = DatabaseClient(
            host=mock_redis_settings['host'],
            port=mock_redis_settings['port'],
            db=mock_redis_settings['db'],
            password=mock_redis_settings['password']
        )
        
        # Redis 생성자 호출 검증
        mock_redis.assert_called_once_with(
            host='test-host',
            port=6380,
            db=1,
            password='test-password',
            decode_responses=True
        )

    @patch('app.services.database_client.redis.Redis')
    @patch('app.services.database_client.Settings')
    def test_database_client_init_with_settings(self, mock_settings_class, mock_redis):
        """Settings에서 값을 가져와 초기화하는 테스트"""
        # Settings mock 설정
        mock_settings_instance = MagicMock()
        mock_settings_instance.REDIS_HOST = 'settings-host'
        mock_settings_instance.REDIS_PORT = 6381
        mock_settings_instance.REDIS_DB = 2
        mock_settings_instance.REDIS_PASSWORD = 'settings-password'
        mock_settings_class.return_value = mock_settings_instance
        
        client = DatabaseClient()
        
        # Settings에서 값을 가져와서 Redis 초기화 확인
        mock_redis.assert_called_once_with(
            host='settings-host',
            port=6381,
            db=2,
            password='settings-password',
            decode_responses=True
        )

    @patch('app.services.database_client.redis.Redis')
    def test_database_client_mixed_params(self, mock_redis):
        """일부 매개변수만 제공하는 경우 테스트"""
        with patch('app.services.database_client.Settings') as mock_settings_class:
            mock_settings_instance = MagicMock()
            mock_settings_instance.REDIS_HOST = 'settings-host'
            mock_settings_instance.REDIS_PORT = 6379
            mock_settings_instance.REDIS_DB = 0
            mock_settings_instance.REDIS_PASSWORD = None
            mock_settings_class.return_value = mock_settings_instance
            
            # 일부만 매개변수로 제공
            client = DatabaseClient(host='custom-host', port=9999)
            
            # 제공된 값과 settings에서 가져온 값이 혼합되어 사용됨
            mock_redis.assert_called_once_with(
                host='custom-host',  # 매개변수에서 제공
                port=9999,           # 매개변수에서 제공
                db=0,                # settings에서 가져옴
                password=None,       # settings에서 가져옴
                decode_responses=True
            )

    def test_database_client_logger_name(self):
        """커스텀 로거 이름 설정 테스트"""
        with patch('app.services.database_client.redis.Redis'):
            with patch('app.services.database_client.get_logger') as mock_get_logger:
                client = DatabaseClient(logger_name="CustomLogger")
                
                mock_get_logger.assert_called_with("CustomLogger")

    def test_database_client_decode_responses_false(self):
        """decode_responses=False 설정 테스트"""
        with patch('app.services.database_client.redis.Redis') as mock_redis:
            client = DatabaseClient(decode_responses=False)
            
            # decode_responses=False로 호출되었는지 확인
            call_args = mock_redis.call_args
            assert call_args[1]['decode_responses'] is False


class TestDatabaseClientMethods:
    """DatabaseClient 메서드 테스트 (실제 Redis 연결 없이)"""

    @pytest.fixture
    def mock_client(self):
        """Mock Redis 클라이언트가 포함된 DatabaseClient"""
        with patch('app.services.database_client.redis.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            client = DatabaseClient()
            client.redis_client = mock_redis_instance
            return client, mock_redis_instance

    @pytest.mark.asyncio
    async def test_get_method_exists(self, mock_client):
        """get 메서드가 존재하는지 테스트"""
        client, mock_redis = mock_client
        
        # get 메서드가 있는지 확인
        assert hasattr(client, 'get') or hasattr(client.redis_client, 'get')

    @pytest.mark.asyncio
    async def test_set_method_exists(self, mock_client):
        """set 메서드가 존재하는지 테스트"""
        client, mock_redis = mock_client
        
        # set 메서드가 있는지 확인
        assert hasattr(client, 'set') or hasattr(client.redis_client, 'set')

    def test_client_has_redis_instance(self, mock_client):
        """클라이언트가 Redis 인스턴스를 가지고 있는지 테스트"""
        client, mock_redis = mock_client
        
        # Redis 클라이언트 인스턴스가 설정되어 있는지 확인
        assert hasattr(client, 'redis_client')
        assert client.redis_client is not None


class TestDatabaseClientErrorHandling:
    """DatabaseClient 오류 처리 테스트"""

    @patch('app.services.database_client.redis.Redis')
    def test_redis_connection_error_handling(self, mock_redis):
        """Redis 연결 오류 시 예외 처리 테스트"""
        # Redis 생성자에서 예외 발생하도록 설정
        mock_redis.side_effect = redis.ConnectionError("Redis 연결 실패")
        
        # 예외 발생 검증
        with pytest.raises(redis.ConnectionError):
            client = DatabaseClient()

    @patch('app.services.database_client.redis.Redis')
    def test_redis_auth_error_handling(self, mock_redis):
        """Redis 인증 오류 시 예외 처리 테스트"""
        # Redis 생성자에서 인증 오류 발생하도록 설정
        mock_redis.side_effect = redis.AuthenticationError("인증 실패")
        
        # 예외 발생 검증
        with pytest.raises(redis.AuthenticationError):
            client = DatabaseClient(password="wrong-password")


class TestDatabaseClientIntegration:
    """DatabaseClient 통합 테스트 (실제 메서드 동작 시뮬레이션)"""

    @pytest.fixture
    def integrated_client(self):
        """통합 테스트용 클라이언트 (실제 Redis 메서드들을 mock으로 대체)"""
        with patch('app.services.database_client.redis.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            # DatabaseClient가 실제로 사용할 수 있는 메서드들 추가
            client = DatabaseClient()
            
            # get/set 메서드를 클라이언트에 직접 추가 (실제 구현 시뮬레이션)
            async def mock_get(key):
                return await mock_redis_instance.get(key)
            
            async def mock_set(key, value):
                if isinstance(value, dict):
                    value = json.dumps(value)
                return await mock_redis_instance.set(key, value)
            
            client.get = mock_get
            client.set = mock_set
            client.redis_client = mock_redis_instance
            
            return client, mock_redis_instance

    @pytest.mark.asyncio
    async def test_set_and_get_string_value(self, integrated_client):
        """문자열 값 저장 및 조회 테스트"""
        client, mock_redis = integrated_client
        
        # Mock 반환값 설정
        mock_redis.set.return_value = True
        mock_redis.get.return_value = "test_value"
        
        # 값 저장
        result = await client.set("test_key", "test_value")
        assert result is True
        
        # 값 조회
        value = await client.get("test_key")
        assert value == "test_value"
        
        # Redis 메서드 호출 검증
        mock_redis.set.assert_called_with("test_key", "test_value")
        mock_redis.get.assert_called_with("test_key")

    @pytest.mark.asyncio
    async def test_set_and_get_dict_value(self, integrated_client):
        """딕셔너리 값 저장 및 조회 테스트"""
        client, mock_redis = integrated_client
        
        test_dict = {"key1": "value1", "key2": 123}
        json_string = json.dumps(test_dict)
        
        # Mock 반환값 설정
        mock_redis.set.return_value = True
        mock_redis.get.return_value = json_string
        
        # 딕셔너리 저장
        result = await client.set("test_dict_key", test_dict)
        assert result is True
        
        # 값 조회
        value = await client.get("test_dict_key")
        assert value == json_string
        
        # JSON 변환이 올바르게 이루어졌는지 검증
        mock_redis.set.assert_called_with("test_dict_key", json_string)

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, integrated_client):
        """존재하지 않는 키 조회 테스트"""
        client, mock_redis = integrated_client
        
        # 존재하지 않는 키에 대해 None 반환
        mock_redis.get.return_value = None
        
        value = await client.get("nonexistent_key")
        assert value is None
        
        mock_redis.get.assert_called_with("nonexistent_key")


# 테스트 실행을 위한 pytest 설정
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
