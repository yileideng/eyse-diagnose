
package com.project.diagnose.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.project.diagnose.pojo.User;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface UserMapper extends BaseMapper<User> {
}

